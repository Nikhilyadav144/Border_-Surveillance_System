"""Screens of the command centre.

Each module here is one navigable page. Pages know about the database and the
configuration, never about Qt threading or OpenCV: the worker hands them
finished results through signals wired up in
:class:`app.gui.main_window.MainWindow`.
"""
