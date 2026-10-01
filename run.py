#!/usr/bin/env python3
"""Start Sarthink: the 3D memory graph + search + memory assistant at http://127.0.0.1:8000

  python3 run.py --demo                 fictional demo archive, CPU only (start here)
  python3 run.py                        your own archive in processed_data/ (needs the index built)
  python3 run.py --gpu HOST PORT        use a remote GPU embed_server over a self-healing SSH tunnel

Without DEEPSEEK_API_KEY in .env the graph, search, stats, inbox and "on this day" still work.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="use the bundled fictional demo archive (demo/)")
    ap.add_argument("--data", help="data folder (default: processed_data, or demo with --demo)")
    ap.add_argument("--model", help="embedding model: 0.6b (CPU) or 8b (GPU); default 0.6b for --demo, else 8b")
    ap.add_argument("--gpu", nargs=2, metavar=("SSH_HOST", "SSH_PORT"), help="tunnel to a vast.ai embed_server")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    env = dict(os.environ)
    env["SARTHINK_DATA"] = a.data or ("demo" if a.demo else env.get("SARTHINK_DATA", "processed_data"))
    env["SARTHINK_EMBED_MODEL"] = a.model or ("0.6b" if a.demo else env.get("SARTHINK_EMBED_MODEL", "8b"))
    data = ROOT / env["SARTHINK_DATA"]
    if not (data / "db" / "sarthink_index.db").exists():
        sys.exit(f"No index found in {data}/db — run with --demo, or build your index first (see README).")

    tunnel = None
    if a.gpu:
        tunnel = subprocess.Popen(["bash", str(ROOT / "scripts" / "api" / "tunnel.sh"), a.gpu[0], a.gpu[1], "8101"])
        env["EMBED_URL"] = "http://127.0.0.1:8101"
        time.sleep(4)
    elif env["SARTHINK_EMBED_MODEL"] == "8b" and not env.get("EMBED_URL"):
        print("Note: the 8b embedding model needs a GPU (--gpu HOST PORT); search falls back to keywords without it.")

    try:
        from dotenv import dotenv_values
        if not (dotenv_values(ROOT / ".env").get("DEEPSEEK_API_KEY") or env.get("DEEPSEEK_API_KEY")):
            print("Note: no DEEPSEEK_API_KEY in .env — the assistant is disabled; graph, search and stats still work.")
    except ImportError:
        sys.exit("Missing dependencies: pip install -r requirements.txt")

    print(f"Sarthink · data: {env['SARTHINK_DATA']} · embeddings: {env['SARTHINK_EMBED_MODEL']}"
          f" · open http://127.0.0.1:{a.port}/  (first start downloads the embedding model)")
    try:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "api" / "server.py"), "--port", str(a.port),
                        "--model", env["SARTHINK_EMBED_MODEL"]], env=env, check=False)
    finally:
        if tunnel:
            tunnel.terminate()


if __name__ == "__main__":
    main()
