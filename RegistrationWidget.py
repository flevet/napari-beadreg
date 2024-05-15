from PyQt5.QtWidgets import QWidget, QPushButton, QSizePolicy, QLabel, QGridLayout, QFileDialog, QGroupBox, QComboBox, QLineEdit, QCheckBox, QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt5.QtCore import QDir
import napari
import numpy as np

from napari.layers import Points, Shapes
from scipy.spatial import KDTree
from utils import ThinPlateSpineTransform, is_palmtracer2_file
import networkx as nx
import random
import math
import sys

source_example = ""
target_example = ""

class RegistrationWidget(QWidget):

    m_filenames = {"source": source_example, "target": target_example}
    m_beads: np.ndarray
    m_nb_beads_source: int
    m_nb_beads_target: int
    m_tree_source: KDTree
    m_tree_target: KDTree
    m_tree: KDTree
    m_bead_layer: Points = None
    m_link_layer: Shapes = None
    m_graph: nx.Graph

    m_dir: str = ""

    m_selectedBead = None
    localizations: np.ndarray = None
    transformed_localizations: np.ndarray = None

    original_width: int = 256
    original_height: int = 256

    m_prevent_update = False

    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        self.viewer = viewer

        self.groupbox_load = QGroupBox("Load Files")
        self.groupbox_load.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.load_source_button = QPushButton("Source data:")
        self.load_source_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.source_name_label = QLabel("")
        self.source_name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.load_target_button = QPushButton("Target data:")
        self.load_target_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.target_name_label = QLabel("")
        self.target_name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.layout_load = QGridLayout()
        self.layout_load.addWidget(self.load_source_button, 0, 0, 1, 1)
        self.layout_load.addWidget(self.source_name_label, 0, 1, 1, 1)
        self.layout_load.addWidget(self.load_target_button, 1, 0, 1, 1)
        self.layout_load.addWidget(self.target_name_label, 1, 1, 1, 1)
        self.groupbox_load.setLayout(self.layout_load)

        self.groupbox_transfo = QGroupBox("Transformation parameters")
        self.groupbox_transfo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.label_radius = QLabel("Search radius (nm)")
        self.label_radius.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.ledit_radius = QLineEdit("50000")
        self.ledit_radius.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.label_multiplier = QLabel("Lateral dimension multiplier")
        self.label_multiplier.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.ledit_multiplier = QLineEdit("1")
        self.ledit_multiplier.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.label_width = QLabel("Original width")
        self.label_width.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.ledit_width = QLineEdit("256")
        self.ledit_width.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.label_height = QLabel("Original height")
        self.label_height.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.ledit_height = QLineEdit("256")
        self.ledit_height.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.label_nb_points = QLabel("# pairs to use")
        self.label_nb_points.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.ledit_nb_points = QLineEdit("3")
        self.ledit_nb_points.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.cbox_nb_points = QCheckBox('Auto')
        self.cbox_nb_points.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.cbox_nb_points.setChecked(True)

        self.layout_transfo = QGridLayout()
        self.layout_transfo.addWidget(self.label_radius, 0, 0, 1, 1)
        self.layout_transfo.addWidget(self.ledit_radius, 0, 1, 1,2)
        self.layout_transfo.addWidget(self.label_multiplier, 1, 0, 1, 1)
        self.layout_transfo.addWidget(self.ledit_multiplier, 1, 1, 1, 2)
        self.layout_transfo.addWidget(self.label_width, 2, 0, 1, 1)
        self.layout_transfo.addWidget(self.ledit_width, 2, 1, 1, 2)
        self.layout_transfo.addWidget(self.label_height, 3, 0, 1, 1)
        self.layout_transfo.addWidget(self.ledit_height, 3, 1, 1, 2)
        self.layout_transfo.addWidget(self.label_nb_points, 5, 0, 1, 1)
        self.layout_transfo.addWidget(self.ledit_nb_points, 5, 1, 1, 1)
        self.layout_transfo.addWidget(self.cbox_nb_points, 5, 2, 1, 1)
        self.groupbox_transfo.setLayout(self.layout_transfo)

        self.groupbox_edition = QGroupBox("Edition")
        self.groupbox_edition.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.button_erase_links = QPushButton("Erase links")
        self.button_erase_links.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.layout_edition = QHBoxLayout()
        self.layout_edition.addWidget(self.button_erase_links)
        self.groupbox_edition.setLayout(self.layout_edition)

        self.groupbox_buttons = QGroupBox("Actions")
        self.groupbox_buttons.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.button_create_transfo = QPushButton("Create transform")
        self.button_create_transfo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.layout_buttons = QGridLayout()
        self.layout_buttons.addWidget(self.button_create_transfo, 0, 0, 1, 1)

        self.load_locs_button = QPushButton("Localization file:")
        self.load_locs_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.locs_name_label = QLabel("")
        self.locs_name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.view_locs_button = QPushButton("View")
        self.view_locs_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.button_apply_transfo = QPushButton("Apply transform")
        self.button_apply_transfo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.button_save_transfo = QPushButton("Save transformed")
        self.button_save_transfo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.transformed_locs_label = QLabel("Transformed locs:")
        self.transformed_locs_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.transformed_name_label = QLabel("")
        self.transformed_name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.view_transformed_locs_button = QPushButton("View")
        self.view_transformed_locs_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.factor_locs_label = QLabel("Factor for saving:")
        self.factor_locs_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.factor_locs_ledit = QLineEdit("1")
        self.factor_locs_ledit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.save_with_factors_button = QPushButton("Save with factor")
        self.save_with_factors_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.layout_buttons.addWidget(self.load_locs_button, 1, 0, 1, 1)
        self.layout_buttons.addWidget(self.locs_name_label, 1, 1, 1, 1)
        self.layout_buttons.addWidget(self.view_locs_button, 1, 2, 1, 1)
        self.layout_buttons.addWidget(self.button_apply_transfo, 0, 1, 1, 1)
        self.layout_buttons.addWidget(self.button_save_transfo, 0, 2, 1, 1)
        self.layout_buttons.addWidget(self.transformed_locs_label, 2, 0, 1, 1)
        self.layout_buttons.addWidget(self.transformed_name_label, 2, 1, 1, 1)
        self.layout_buttons.addWidget(self.view_transformed_locs_button, 2, 2, 1, 1)
        self.layout_buttons.addWidget(self.factor_locs_label, 3, 0, 1, 1)
        self.layout_buttons.addWidget(self.factor_locs_ledit, 3, 1, 1, 1)
        self.layout_buttons.addWidget(self.save_with_factors_button, 3, 2, 1, 1)
        self.groupbox_buttons.setLayout(self.layout_buttons)

        self.empty_widget = QWidget()
        self.empty_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.groupbox_load)
        self.layout.addWidget(self.groupbox_transfo)
        self.layout.addWidget(self.groupbox_edition)
        self.layout.addWidget(self.groupbox_buttons)
        self.layout.addWidget(self.empty_widget)

        self.load_source_button.clicked.connect(self.open_localization_file)
        self.load_target_button.clicked.connect(self.open_localization_file)
        self.button_create_transfo.clicked.connect(self.create_transformation)
        self.button_erase_links.clicked.connect(self.erase_links)
        self.load_locs_button.clicked.connect(self.open_localization_file)
        self.button_apply_transfo.clicked.connect(self.apply_transformation)
        self.button_save_transfo.clicked.connect(self.save_transformed_locs)
        self.save_with_factors_button.clicked.connect(self.save_transformed_locs_with_factor)

        self.view_locs_button.clicked.connect(self.view_points_layer)
        self.view_transformed_locs_button.clicked.connect(self.view_points_layer)

        self.groupbox_transfo.setVisible(True)
        self.groupbox_buttons.setVisible(False)

        self.setLayout(self.layout)

        if source_example != "" and target_example != "":
            self.load_beads()

    def open_localization_file(self):
        path = QDir.currentPath() if self.m_dir == ""  else self.m_dir
        filename = QFileDialog.getOpenFileName(None, 'Select localization file', path, 'Localization file (*.txt *.csv)', options=QFileDialog.DontUseNativeDialog)
        if filename[0] == "":
            return
        short_name = filename[0].split('/')[-1]
        self.m_dir = filename[0].replace(short_name, "")
        sender = self.sender()
        if sender is self.load_source_button:
            self.m_filenames["source"] = filename[0]
            self.source_name_label.setText(short_name)
        if sender is self.load_target_button:
            self.m_filenames["target"] = filename[0]
            self.target_name_label.setText(short_name)
        if sender is self.load_locs_button:
            self.m_filenames["localization"] = filename[0]
            self.locs_name_label.setText(short_name)
            localizations = list()
            self.load_bead_file(self.m_filenames['localization'], localizations)
            self.localizations = np.asarray(localizations)
            self.transformed_localizations = None
            self.transformed_name_label.setText('')
            return

        if self.m_filenames['source'] != "" and self.m_filenames['target'] != "":
            self.load_beads()

    def load_bead_file(self, filename: str, beads: list):
        multiplier: float = float(self.ledit_multiplier.text())
        if is_palmtracer2_file(filename):
            self.load_palmtracer2_file(filename, beads, multiplier)
        else:
            self.load_thunderstorm_file(filename, beads, multiplier)

    def load_palmtracer2_file(self, filename: str, beads: list, multiplier: float):
        with open(filename) as f:
            lines = [line.rstrip() for line in f]
        columns = lines[1].split('\t')
        self.original_width = int(int(columns[0]) * multiplier)
        self.original_height = int(int(columns[1]) * multiplier)
        pixel_size = float(columns[4]) * 1000. #change pixel size to nm
        self.ledit_width.setText(str(self.original_width))
        self.ledit_height.setText(str(self.original_height))
        for n in range(3, len(lines)):
            columns = lines[n].split('\t')
            beads.append([float(columns[5]) * pixel_size * multiplier, float(columns[6]) * pixel_size * multiplier])
        return

    def load_thunderstorm_file(self, filename: str, beads: list, multiplier: float):
        with open(filename) as f:
            lines = [line.rstrip() for line in f]
        for n in range(1, len(lines)):
            columns = lines[n].split(',')
            x, y = float(columns[2]) * multiplier, float(columns[3]) * multiplier
            beads.append([x, y])
            if x > self.original_width:
                self.original_width = int(x)
            if y > self.original_height:
                self.original_height = int(y)
        self.ledit_width.setText(str(self.original_width))
        self.ledit_height.setText(str(self.original_height))
        return

    def load_beads(self):
        beads_source, beads_target = list(), list()
        self.load_bead_file(self.m_filenames['source'], beads_source)
        self.m_nb_beads_source = len(beads_source)
        self.load_bead_file(self.m_filenames['target'], beads_target)
        self.m_nb_beads_target = len(beads_target)
        self.m_beads = np.asarray(beads_source + beads_target)
        self.m_tree_source = KDTree(np.asarray(beads_source))
        self.m_tree_target = KDTree(np.asarray(beads_target))
        print(self.m_beads)

        self.m_tree = KDTree(np.asarray(self.m_beads))

        color_source = [[1, 0, 0] for i in range(self.m_nb_beads_source)]
        color_target = [[1, 1, 0] for i in range(self.m_nb_beads_target)]
        colors = np.asarray(color_source + color_target)

        beads_text = ['source_{}'.format(i + 1) for i in range(self.m_nb_beads_source)] + ['target_{}'.format(i + 1) for i in range(self.m_nb_beads_target)]
        features = {
            'type': beads_text,
        }

        dim_bigger = self.original_width if self.original_width > self.original_height else self.original_height
        multiplier_rendering = 1 if dim_bigger <= 256. else float(dim_bigger) / 256.

        if self.m_bead_layer is None:
            self.m_bead_layer = self.viewer.add_points(self.m_beads, face_color=colors, size=5*multiplier_rendering, features=features, name='Beads')
        else:
            self.m_bead_layer.data = self.m_beads
            self.m_bead_layer.face_color = colors

        self.groupbox_transfo.setVisible(True)
        self.groupbox_buttons.setVisible(True)

        radius = float(self.ledit_radius.text())
        neighbors_source_distance, neighbors_source_indexes = self.m_tree_target.query(self.m_beads[:self.m_nb_beads_source], 1)
        neighbors_target_distance, neighbors_target_indexes = self.m_tree_source.query(self.m_beads[-self.m_nb_beads_target:], 1)

        self.m_graph = nx.Graph()
        self.m_graph.add_nodes_from(np.arange(self.m_nb_beads_source + self.m_nb_beads_target))
        for index_source, index_target, distance in zip(np.arange(self.m_nb_beads_source), neighbors_source_indexes, neighbors_source_distance):
            if distance < radius:
                self.m_graph.add_edge(index_source, self.m_nb_beads_source + index_target)
        for index_source, index_target, distance in zip(neighbors_target_indexes, np.arange(self.m_nb_beads_target), neighbors_target_distance):
            if distance < radius:
                self.m_graph.add_edge(index_source, self.m_nb_beads_source + index_target)

        #Determine the general direction of the links
        vectors = []
        for edge, index in zip(list(self.m_graph.edges), range(self.m_graph.number_of_edges())):
            v = self.m_beads[edge[1]] - self.m_beads[edge[0]]
            vectors.append([v[0], v[1], index])

        sorted(vectors, key=self.clockwiseangle_and_distance)

        median_v = vectors[int(len(vectors) / 2)]
        median_angle, _ = self.clockwiseangle_and_distance(median_v)

        #determine which vectors are far from the median one and tag them
        limit_angle = math.radians(60)
        tagged_edges = [False] * self.m_graph.number_of_edges()
        edges_as_list = list(self.m_graph.edges)
        for edge, index in zip(edges_as_list, range(self.m_graph.number_of_edges())):
            angle, d = self.clockwiseangle_and_distance(self.m_beads[edge[1]] - self.m_beads[edge[0]])
            if math.fabs(median_angle - angle) > limit_angle:
                print('Tag this edge ' + str(index))
                id = edges_as_list.index(tuple(sorted(edge)))
                tagged_edges[id] = True

        #determine which nodes is connected to more than one and tag the longer edges
        for node in self.m_graph.nodes:
            edges = list(self.m_graph.edges(node))
            #determine the true number of edges as we don't count the one tagged just above
            kept_edges = []
            for edge in edges:
                id = edges_as_list.index(tuple(sorted(edge)))
                if tagged_edges[id] is False:
                    kept_edges.append(edge)
            if len(kept_edges) > 1:
                d, e = sys.float_info.max, None
                for edge in kept_edges:
                    dtmp = math.dist(self.m_beads[edge[0]], self.m_beads[edge[1]])
                    if dtmp < d:
                        d = dtmp
                        e = edge
                for edge in kept_edges:
                    if edge != e:
                        id = edges_as_list.index(tuple(sorted(edge)))
                        print('Tag this edge ' + str(edge) + ", " + str(id))
                        tagged_edges[id] = True

        self.updateGraph(tagged_edges)

    def clockwiseangle_and_distance(self, vector):
        refvec = [0, 1]
        # Length of vector: ||v||
        lenvector = math.hypot(vector[0], vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -math.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0] / lenvector, vector[1] / lenvector]
        dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]  # x1*x2 + y1*y2
        diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]  # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to subtract them
        # from 2*pi (360 degrees)
        if angle < 0:
            return 2 * math.pi + angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance should come first.
        return angle, lenvector

    def updateGraph(self, tagged_edges = None):
        if len(self.m_graph.edges) == 0:
            return
        edges, text = list(), list()
        for edge in list(self.m_graph.edges):
            edges.append(np.array([self.m_beads[edge[0]], self.m_beads[edge[1]]]))
            text.append('{},{}'.format(edge[0], edge[1]))

        features = {
            'link': text,
        }

        dim_bigger = self.original_width if self.original_width > self.original_height else self.original_height
        multiplier_rendering = 1 if dim_bigger <= 256. else float(dim_bigger) / 256.
        if self.m_link_layer is None:
            self.m_link_layer = self.viewer.add_shapes(edges, shape_type='line', edge_width=1*multiplier_rendering, edge_color='coral', name='Links', features=features)
            self.m_link_layer.events.set_data.connect(self.update_graph_on_layer_event)
        else:
            self.m_prevent_update = True
            self.m_link_layer.data = edges
            self.m_link_layer.features = features
            self.m_link_layer.edge_width = [multiplier_rendering] * len(edges)
            self.m_prevent_update = False
        if self.cbox_nb_points.isChecked():
            self.ledit_nb_points.setText(str(len(edges)))

        if tagged_edges is not None:
            self.m_link_layer.mode = 'direct'
            indexes = []
            for n in range(len(tagged_edges)):
                if tagged_edges[n] is True:
                    indexes.append(n)
            self.m_link_layer.selected_data = indexes

    def create_transformation(self):
        for node in np.arange(self.m_nb_beads_source + self.m_nb_beads_target):
            if self.m_graph.degree[node] > 1:
                mess = QMessageBox()
                if node < self.m_nb_beads_source:
                    mess.setText('The source bead ' + str(node) + ' is connected to more than one bead. Please correct before proceeding.')
                else:
                    mess.setText('The source bead ' + str(node - self.m_nb_beads_source) + ' is connected to more than one bead. Please correct before proceeding.')
                mess.exec()
                return

        #Get params
        edges = list(self.m_graph.edges)
        nb_points = len(edges) if self.cbox_nb_points.isChecked() else int(self.ledit_nb_points.text())
        dim_bigger = self.original_width if self.original_width > self.original_height else self.original_height
        multiplier_rendering = 1 if dim_bigger <= 256. else float(dim_bigger) / 256.
        sources, targets = list(), list()
        if nb_points != len(edges):
            randomlist = random.sample(range(0, len(edges)), nb_points)
            for index in randomlist:
                sources.append(self.m_beads[edges[index][0]])
                targets.append(self.m_beads[edges[index][1]])
        else:
            for edge in edges:
                sources.append(self.m_beads[edge[0]])
                targets.append(self.m_beads[edge[1]])

        self.registration = ThinPlateSpineTransform(X=np.asarray(targets), Y=np.asarray(sources))
        test_transformed = self.registration.transform_point_cloud(self.m_beads[:self.m_nb_beads_source].astype(np.float32))
        layer = self.viewer.add_points(test_transformed, face_color='magenta', size=5*multiplier_rendering, name='Transformed beads')

        #create lines for showing deformation
        nb_lines_x: int = 26
        nb_lines_y: int = 26
        step_x, step_y = float(self.original_width) / float(nb_lines_x - 1), float(self.original_height) / float(nb_lines_y - 1)
        lines = list()
        for x in np.arange(0, float(self.original_width + step_x), step_x):
            line = list()
            for y in np.arange(0, float(self.original_height + step_y), step_y):
                line.append(np.array([x, y], dtype=float))
            lines.append(np.asarray(line))
        for y in np.arange(0, float(self.original_height + step_y), step_y):
            line = list()
            for x in np.arange(0, float(self.original_width + step_x), step_x):
                line.append(np.array([x, y], dtype=float))
            lines.append(np.asarray(line))

        transformed_lines = list()
        for line in lines:
            transformed_lines.append(self.registration.transform_point_cloud(line))

        test_trans = self.viewer.add_shapes(np.asarray(transformed_lines), shape_type='path', edge_width=0.5*multiplier_rendering, edge_color='white')


    def cancel_bead_selection(self):
        self.m_selectedBead = None

    def selectBead(self, bead):
        if self.m_selectedBead is None:
            self.m_selectedBead = bead
            return False
        else:
            type1, type2 = self.m_selectedBead[1][0].split("_")[0], bead[1][0].split("_")[0]
            if type1 == type2:
                return
            self.m_graph.remove_edges_from(list(self.m_graph.edges(self.m_selectedBead[0])))
            self.m_graph.remove_edges_from(list(self.m_graph.edges(bead[0])))
            self.m_graph.add_edge(self.m_selectedBead[0], bead[0])
            self.m_selectedBead = None
            return True


    def erase_links(self):
        if len(self.m_link_layer.selected_data) == 0:
            return
        for selection in self.m_link_layer.selected_data:
            print('{}, {}'.format(selection, self.m_link_layer.features.values[selection]))
            nodes = self.m_link_layer.features.values[selection][0].split(',')
            self.m_graph.remove_edge(int(nodes[0]), int(nodes[1]))
        self.updateGraph()

    def apply_transformation(self):
        if self.localizations is not None:
            self.transformed_localizations = self.registration.transform_point_cloud(self.localizations.astype(np.float32))
            self.transformed_name_label.setText('transformed_' + self.locs_name_label.text())

    def view_points_layer(self):
        sender = self.sender()
        if sender is self.view_locs_button:
            if self.localizations is not None:
                layer = self.viewer.add_points(self.localizations, face_color='red', size=1, edge_width=0, name='Localizations')
        elif sender is self.view_transformed_locs_button:
            if self.transformed_localizations is not None:
                layer = self.viewer.add_points(self.transformed_localizations, face_color='magenta', size=1, edge_width=0, name='Transformed localizations')

    def save_transformed_locs(self):
        if self.m_filenames["localization"] != '' and self.transformed_localizations is not None:
            list_strs = self.m_filenames["localization"].split('/')
            list_strs[-1] = self.transformed_name_label.text()
            filename = '/'.join(list_strs)
            print(filename)

            if is_palmtracer2_file(self.m_filenames["localization"]):
                with open(self.m_filenames["localization"]) as f_locs, open(filename, 'w') as f_transformed:
                    lines = [line.rstrip() for line in f_locs]
                    for n in range(0, 3):
                        f_transformed.write(lines[n] + '\n')
                    columns = lines[1].split('\t')
                    pixel_size = float(columns[4]) * 1000.  # change pixel size to nm
                    for n in range(3, len(lines)):
                        columns = lines[n].split('\t')
                        index_loc = n - 3
                        columns[5] = str(self.transformed_localizations[index_loc][0] / pixel_size)
                        columns[6] = str(self.transformed_localizations[index_loc][1] / pixel_size)
                        f_transformed.write('\t'.join(columns) + '\n')
            else:
                with open(self.m_filenames["localization"]) as f_locs, open(filename, 'w') as f_transformed:
                    lines = [line.rstrip() for line in f_locs]
                    f_transformed.write(lines[0] + '\n')
                    for n in range(1, len(lines)):
                        columns = lines[n].split(',')
                        index_loc = n - 1
                        columns[2] = str(self.transformed_localizations[index_loc][0])
                        columns[3] = str(self.transformed_localizations[index_loc][1])
                        f_transformed.write(','.join(columns) + '\n')

    def save_transformed_locs_with_factor(self):
        if self.m_filenames["localization"] == '' or self.transformed_localizations is None:
            mess = QMessageBox()
            mess.setText('Localizations have not been transformed')
            mess.exec()
            return

        factor = float(self.factor_locs_ledit.text())

        if is_palmtracer2_file(self.m_filenames["localization"]):
            mess = QMessageBox()
            mess.setText('This button should be only used if your original localization file was in nm (i.e. from Thunderstorm.')
            mess.exec()
            return

        list_strs = self.m_filenames["localization"].split('/')
        list_strs[-1] = 'scaled_' + self.transformed_name_label.text()
        filename = '/'.join(list_strs)
        print(filename)

        with open(self.m_filenames["localization"]) as f_locs, open(filename, 'w') as f_transformed:
            lines = [line.rstrip() for line in f_locs]
            f_transformed.write(lines[0] + '\n')
            for n in range(1, len(lines)):
                columns = lines[n].split(',')
                index_loc = n - 1
                x, y = self.transformed_localizations[index_loc][0], self.transformed_localizations[index_loc][1]
                columns[2] = str(x * factor)
                columns[3] = str(y * factor)
                f_transformed.write(','.join(columns) + '\n')

    def update_graph_on_layer_event(self):
        if self.m_prevent_update:
            return
        if len(self.m_graph.edges) != len(self.m_link_layer.data):
            self.m_graph.remove_edges_from(list(self.m_graph.edges()))
            for edge in self.m_link_layer.features['link']:
                nodes = edge.split(',')
                self.m_graph.add_edge(int(nodes[0]), int(nodes[1]))