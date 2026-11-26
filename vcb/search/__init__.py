"""Optional online search (off by default).

Importing this package is free: it pulls in nothing heavy and touches no
network. `requests`/PySocks are imported only by `vcb.search.net`, which is
loaded only when the user has turned search on.
"""
