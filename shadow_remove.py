import os
import glob
from ShadowRemoval import *

if not os.path.exists("shadow-adjusted"):
    os.makedirs("shadow-adjusted")

if not os.path.exists("shadow-adjusted/normal"):
    os.makedirs("shadow-adjusted/normal")

if not os.path.exists("shadow-adjusted/shadows-obscure-visibility"):
    os.makedirs("shadow-adjusted/shadows-obscure-visibility")

for x in os.listdir("confirmed/normal"):
    ShadowRemover.removeShadows("confirmed/normal/" + x, "shadow-adjusted/normal/" + x)

for x in os.listdir("confirmed/shadows-obscure-visibility/"):
    ShadowRemover.removeShadows("confirmed/shadows-obscure-visibility/" + x, "shadow-adjusted/shadows-obscure-visibility/" + x)
