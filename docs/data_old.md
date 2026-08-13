Two-layer flow phantom DCS dataset
This dataset contains the primary research data used in the analysis presented in the paper entitled "Deep-learning-based separation of shallow and deep layer blood flow rates in diffuse correlation spectroscopy," by Nakabayashi, M., Liu, S., Broti, N.M., Ichinose, M., and Ono, Y. (2023), published in Biomedical Optics Express.



The dataset contains 1540 autocorrelation (g2) functions collected at three different source-detector separations (1, 2, 3 cm) in diffuse correlation spectroscopy (DCS). Each function represents the averaged waveform of stable observations over 60 instances, corresponding to one minute of measurement time. Measurements were made using a two-layer flow phantom.



Each data set consists of three autocorrelation functions, the preset and measured flow rate combinations, and the estimated Blood Flow Index (BFI) values calculated using the homogeneous semi-infinite plate medium model. The time legend of the autocorrelation function is 2.5e-6:2.5e-6:5e-4 seconds.



The dataset is in MATLAB *.mat file format and includes two components: 



Dataset_set_intra40 - a 7 x 1540 matrix with the following rows: 

Row 1 - Shallow layer flow rate set at the pump 
Row 2 - Deep layer flow rate set at the pump 
Row 3 - Shallow layer flow rate measured at the phantom outlet 
Row 4 - Deep layer flow rate measured at the phantom outlet 
Row 5 - BFI estimated with DCS of 1 cm source-detector separation 
Row 6 - BFI estimated with DCS of 2 cm source-detector separation 
Row 7 - BFI estimated with DCS of 3 cm source-detector separation
ave_gtau_intra40 - a 10 x 501 x 1540 matrix, the first 3 x 501 data points represent autocorrelation functions obtained with DCS of 1, 2, and 3 cm source-detector separations. The length of each autocorrelation function is 501. The information in the subsequent 4 to 10 rows mirrors that of Dataset_set_intra40. In total, 3 x 1540 autocorrelation functions are recorded. 