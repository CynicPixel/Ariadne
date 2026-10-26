import runpy, sys, types
try:
    import PIL.Image
    print("PIL", PIL.__version__, flush=True)
except ImportError:
    import cv2
    class _Img:
        def __init__(self, a): self.a = a
        def save(self, p): cv2.imwrite(str(p), cv2.cvtColor(self.a, cv2.COLOR_RGB2BGR))
    m = types.ModuleType("PIL"); m.Image = types.SimpleNamespace(fromarray=_Img)
    sys.modules["PIL"], sys.modules["PIL.Image"] = m, m.Image
    print("PIL absent: cv2-backed Image.fromarray(...).save", flush=True)
sys.argv = ["train_worldgen_gsplat.py", "train", "--data", "/data", "--out", "/out", "--steps", sys.argv[1]]
runpy.run_path("/data/train_worldgen_gsplat.py", run_name="__main__")
