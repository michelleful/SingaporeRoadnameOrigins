SingaporeRoadnameOrigins
========================

A map of Singapore, with roads colour-coded for language of origin

Plan:
-----

- [x] Get list of Singapore roadnames
- [ ] Use scikit-learn to classify roadnames into languages of origin, which involves:
- [x] Extract training features
- [ ] Experiment with various machine learning algorithms
- [ ] Evaluate results
- [ ] Iteratively perform manual correction and expand the training set
- [ ] Use OpenStreetMap and GeoJSON (?) to plot the map

Optional:

- [ ] Use GeoDjango to produce interactive database 
- [ ] Compare roadnames in CBD to the Jackson town plan of 1822, which
    segregated races into different zones
- [ ] Devise an evaluation metric to measure the homogeneity of roadnames
    within a certain area
- [ ] Apply evaluation metric to data such as electoral constituency boundaries
    as they evolve over time

Community contributions:

- [ ] Fix roadnames with spelling errors in OpenStreetMap
- [ ] Roadnames classified by language of origin, possibly with
    community-contributed notes?
- [ ] (optional) map layers for historical electoral constituency boundaries
