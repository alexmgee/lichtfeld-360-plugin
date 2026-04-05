# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin entrypoint for 360 Plugin."""

import lichtfeld as lf

from .panels import Plugin360Panel

_CLASSES = [Plugin360Panel]


def on_load():
    for cls in _CLASSES:
        lf.register_class(cls)
    lf.ui.set_panel_space(Plugin360Panel.id, lf.ui.PanelSpace.MAIN_PANEL_TAB)
    lf.ui.set_panel_order(Plugin360Panel.id, Plugin360Panel.order)
    lf.log.info("360 Plugin loaded")


def on_unload():
    for cls in reversed(_CLASSES):
        lf.unregister_class(cls)
    lf.log.info("360 Plugin unloaded")
