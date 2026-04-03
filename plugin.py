# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin entrypoint for PanoSplat."""

import lichtfeld as lf

from .panels import PanoSplatPanel

_CLASSES = [PanoSplatPanel]


def on_load():
    for cls in _CLASSES:
        lf.register_class(cls)
    lf.ui.set_panel_space(PanoSplatPanel.id, lf.ui.PanelSpace.MAIN_PANEL_TAB)
    lf.ui.set_panel_order(PanoSplatPanel.id, PanoSplatPanel.order)
    lf.log.info("PanoSplat plugin loaded")


def on_unload():
    for cls in reversed(_CLASSES):
        lf.unregister_class(cls)
    lf.log.info("PanoSplat plugin unloaded")
