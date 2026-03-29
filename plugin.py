# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin entrypoint for the 360 Camera plugin."""

import lichtfeld as lf

from .panels import Prep360Panel

_CLASSES = [Prep360Panel]


def on_load():
    for cls in _CLASSES:
        lf.register_class(cls)
    lf.ui.set_panel_space(Prep360Panel.id, lf.ui.PanelSpace.MAIN_PANEL_TAB)
    lf.ui.set_panel_order(Prep360Panel.id, Prep360Panel.order)
    lf.log.info("360 Camera plugin loaded")


def on_unload():
    for cls in reversed(_CLASSES):
        lf.unregister_class(cls)
    lf.log.info("360 Camera plugin unloaded")
