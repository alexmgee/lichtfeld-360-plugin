# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin entrypoint for 360 Plugin."""

import lichtfeld as lf

from .panels import Plugin360Panel

_CLASSES = [Plugin360Panel]


def _register_class(cls):
    register = getattr(lf, "register_class", None)
    if callable(register):
        register(cls)
        return
    ui_register = getattr(getattr(lf, "ui", None), "register_class", None)
    if callable(ui_register):
        ui_register(cls)
        return
    raise AttributeError("No register_class function available on lichtfeld or lichtfeld.ui")


def _unregister_class(cls):
    unregister = getattr(lf, "unregister_class", None)
    if callable(unregister):
        unregister(cls)
        return
    ui_unregister = getattr(getattr(lf, "ui", None), "unregister_class", None)
    if callable(ui_unregister):
        ui_unregister(cls)
        return
    raise AttributeError("No unregister_class function available on lichtfeld or lichtfeld.ui")


def on_load():
    for cls in _CLASSES:
        _register_class(cls)
    set_panel_space = getattr(lf.ui, "set_panel_space", None)
    set_panel_order = getattr(lf.ui, "set_panel_order", None)
    panel_space = getattr(
        lf.ui.PanelSpace,
        "MAIN_PANEL_TAB",
        getattr(lf.ui.PanelSpace, "FLOATING", None),
    )
    if callable(set_panel_space) and panel_space is not None:
        set_panel_space(Plugin360Panel.id, panel_space)
    if callable(set_panel_order):
        set_panel_order(Plugin360Panel.id, Plugin360Panel.order)
    lf.log.info("360 Plugin loaded")


def on_unload():
    for cls in reversed(_CLASSES):
        _unregister_class(cls)
    lf.log.info("360 Plugin unloaded")
