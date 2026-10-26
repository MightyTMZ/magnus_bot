"""Vision: reading the physical board.

The camera (an iPhone via iVCam) and the RF-DETR detector run as a separate process,
``RF-DETR/vision_server.py``, in RF-DETR's own venv: its torch 2.14 + CUDA cannot share an
environment with lerobot's ``torch<2.12``. That server publishes per-square occupancy over
localhost. This package is the core side and stays stdlib-only:

* :mod:`~magnus_bot.vision.observation`: ``BoardObservation`` and ``HttpBoardReader``.
* :mod:`~magnus_bot.vision.verify`: grade a transfer against the camera (``check_transfer``).
* :mod:`~magnus_bot.vision.sim_board`: ``SimulatedBoard``, the camera stand-in for sim runs.

:mod:`~magnus_bot.vision.camera` and :mod:`~magnus_bot.vision.homography` are the original
in-process stubs; the server (with ``RF-DETR/board.py``) does that work today.
"""
