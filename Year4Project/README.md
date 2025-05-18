# Year4Project
## Summary
This project was to evaluate the success of membership inference attacks against time series synthetic medical data. This successfully worked, and it was possible to reduce MIA success to 0.5 (optimal), as well as the project showing that there is significant risk of MIA against time-series data.

If you have any questions or would like to read the report, please email: ***toby@redshaw.me***

## Files
Navigate to *mia_against_time_series*. 
* **MIA**: Main file to run the MIA process. 
* **Evaluator**: DOMIAS method to evaluate success of MIA. Note, empty functions for future methods.
* **Generator**: Implement synthetic data synthesisers here.
* **Plotter**: Used to plot the results.

## Loading Data
Within *mia.py*, load the data into path. Within this project, the format was each row being a singular ECG waveform, each cut to 488 points. All data signals must be the same length.

## Abstract
Data empowers decision making and research every day, however a significant amount of it is private. Synthetic data is a novel approach to sharing personal information without compromising privacy; within the healthcare sector the sharing of data will create more accurate and faster diagnosis methods, resulting in better patient outcomes. Recent studies show that synthetic data is not as private as initially thought, with membership inference attacks (MIA) being able to deduce if an individual was in the training dataset of a generative model (GM) that produced synthetic data. An attacker could then release the findings of their attack with malicious intent. MIAs have only been conducted against tabular data, and with increasing research being done into time-series GMs, there is a need to quantify the success of MIAs against time-series data. 

The aim of this project was to create a novel MIA method against ECG data and test preventative solutions. It was found that an increase in training data and a low number of epochs could reduce MIA success to 0.5 AUC, the minimum risk possible. However, no strong conclusions were made regarding the effect of the quantity of synthetic data released. These solutions provide promise for the future protection of shared data, however, research must be continuously conducted into MIAs against time-series data due to limitations shown by the MIA method used. The suggested future improvements will help provide a better understanding on how to reduce the risk of MIAs in the future.