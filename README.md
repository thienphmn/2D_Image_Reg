The used datasets are the following:

Paired CT and MRI Dataset for Medical Applications: https://www.kaggle.com/datasets/29c3607295965ebb030f2d158fec487412d84c82528dd44f8ef956aef35541aa

QIN-Breast: https://www.cancerimagingarchive.net/collection/qin-breast/

Preprocessing for "Paired CT and MRI Dataset for Medical Applications":
- padding
- intensity normalization
- gaussian blurring on both CT and MRI data 

Preprocessing for "QIN-Breast":
- intensity normalization
- gaussian blurring
- histogram equalization on both CT and PET data
- for testing 2D rigid registration, corresponding slices were taken out of both CT and PET images to perform the registration algorithm on