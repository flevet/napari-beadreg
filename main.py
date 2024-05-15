import napari
from RegistrationWidget import RegistrationWidget

viewer = napari.Viewer()
dock_widget = RegistrationWidget(viewer)
viewer.window.add_dock_widget(dock_widget)

@viewer.mouse_drag_callbacks.append
def get_event(viewer, event):
    # on press
    yield

    # on release
    for layer in viewer.layers.selection:
        if layer is dock_widget.m_bead_layer:
            if len(layer.selected_data) == 0:
                dock_widget.cancel_bead_selection()
            elif len(layer.selected_data) == 1:
                index = list(layer.selected_data)[0]
                if dock_widget.selectBead((index, layer.features.values[index])):
                    layer.selected_data = []
                    dock_widget.updateGraph()
            else:
                return

napari.run()