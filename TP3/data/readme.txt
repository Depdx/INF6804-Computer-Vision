Le résultat produit par votre méthode de suivi devrait être fourni dans un
fichier 'results.txt', et devrait être formaté afin que nous puissions le
lire automatiquement. Nous nous attendons à la structure suivante:

 <frameid> <cupid> <xmin> <ymin> <width> <height>


Exemple: si pour la 458e trame de la séquence votre méthode détermine que
la tasse 1 (en rouge) est dans le rectangle définit par deux points <x0,y0> [1149,273] et <x1,y1> [1639,706], et que la tasse 2 (en blanc avec des quadrillages) est dans le rectangle définit par <x0,y0> [441,333] et <x1,y1> [1186,967], alors la sortie sous format <x,y,w,h> pour la ligne équivalente dans 'results.txt' sera:

 458 1 1149 273 490 433
 458 2 441 333 745 634

La première ligne devrait posséder un index de trame de 1. Notez par ailleurs que la boîte donnée pour
l'initialisation de votre méthode dans 'init.txt' est donnée sous ce format.

----------------------

The output of your tracker should be provided in a 'results.txt' file,
and should be formatted so that we can read it automatically. We expect
the following structure for our parser:

 <frameid> <cupid> <xmin> <ymin> <width> <height>

Example: if for the 458th frame of the sequence your method says the cup1 (in red)
is in the rectangle defined with two points <x0,y0> [1149,273] and <x1,y1> [1639,706], and the cup2 (in white with lines) is in the rectangle defined with <x0,y0> [441,333] and <x1,y1> [1186,967],
then the output in the format <x,y,w,h> for that line in 'results.txt' should be:

 458 1 1149 273 490 433
 458 2 441 333 745 634

The first line should start with a frame index of 1. Besides, note that the bounding box provided for
the initialization of your method in 'init.txt' is formatted the same way.
