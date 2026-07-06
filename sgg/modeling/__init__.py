__all__ = ["SceneGraphDetector"]


def __getattr__(name):
    if name == "SceneGraphDetector":
        from .detectors import SceneGraphDetector

        return SceneGraphDetector
    raise AttributeError(name)
