"""Text-to-speech benchmark harness for VCB.

Entry point: tools/tts_benchmark.py. The package is stdlib-only, except for the
engines themselves, which run in their own worker processes (worker.py).
Keep this file free of imports: the worker is started as `-m tts_bench.worker`
inside engine environments that may be older than the benchmark's own Python.
"""
