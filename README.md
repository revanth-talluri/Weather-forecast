# Weather-forecast
Predicting the weather based on past data and checking the variations with the true values.

The data for this is taken from UCI ml repo. You can check the data description and download the data from [here](https://archive.ics.uci.edu/ml/datasets/Bias+correction+of+numerical+prediction+model+temperature+forecast).

## Dataset Information
This data is for the purpose of bias correction of next-day maximum and minimum air temperatures forecast of the LDAPS model operated by the Korea Meteorological Administration over Seoul, South Korea. This data consists of summer data from 2013 to 2017. The input data is largely composed of the LDAPS model's next-day forecast data, in-situ maximum and minimum temperatures of present-day, and geographic auxiliary variables. There are two outputs (i.e. next-day maximum and minimum air temperatures) in this data.
Source: https://archive.ics.uci.edu/ml/datasets/Bias+correction+of+numerical+prediction+model+temperature+forecast

There are a total of 7752 rows and 25 columns in the dataset. Some of the columns include station, date, latitude, longitude, present max and min temperature, next day max and min temperature, amount of radiation received, slope of the place etc. The last 2 rows of the dataset has missing values of date and station info. So, let's remove those 2 rows from the dataset. 

There are 2 steps involved here:
- Data Visualization
- Forecasting and checking the bias

## Data Visualization
The date column is split into Year, Month and Day. Check the dataframe info and you can see that there are null values in all columns except the 'station' and 'Date' columns. We can also observe that all the columns can be grouped into 3 groups based on the no. of missing values in their respective columns. Upon further inspection, we can see that these missing values are at the same indices for those particular groups.

We have a total of 25 stations in the data, i.e each station has 310 data points spread evenly across 5 years and between the months of June and August. Let's divide all the data according to the stations. Create 26 empty lists to store the stations data .Note here that we need only 25 empty lists to store the data of 25 stations. But I am creating 26 lists and will leave the 1st list (index=0) empty, so that if I need the data for station i, I can access directly station[i] and get the data. This is done to avoid any confusion. All the columns are of datatype float64 and since the std. deviaiton is less and the total no. of missing values are less, fill all the nan values with the respective column avg.

Plot the geographical locations of the stations from the latitude, longitude data to get a rough idea whethee they're too close and this can help to check if all the stations are independent of each other or if there are any correlations. The plot shows that the stations are located at far enough distance w.r.t each other and can be assumed to be independent of each other. 

Though this is a time-series data, the temperature of the day depends on the meterological factors on that particular day. And if we can measure those factors for the following day, we can predict the temperature on the next day. Let's see the variation of Present_Max temperature for 4 years (2013 to 2016) at all stations and in the later case, the variation of Present_Max temperature at all stations in a particular year. In the first case, we see that there is no relation between the Present_Max temperature when checked throughout years at a particular station. But in the later case, we can see there is a similar trend of the temperature across the stations every year.

That's all with the data visualization and let's move onto the forecasting step.

## Forecasting and checking the bias
Let's use the data from 2013 to 2016 as training data and forecast the 2017 weather data. We will use the 'Multivariate Linear Regression' model to forecast. And in this model, we will not fit the model on all the data between 2013 and 2016, but we will do it seperately for each station, since we have established that they are independent of each other and this can help to avoid the bias from the other 24 stations at that particular station. For each station we have 310 data points, and out of these 248 data points are used for training and 62 for forecasting. Of the 25 columns, some of the data is particular to the station and we will remove such columns. They are 'Date', 'station', 'lat', 'lon', 'DEM', 'Slope', 'Year', 'Month', 'Day'. With the remaining the columns we will train the 'Multivariate Regression' model to predict the 'Next_Tmax' and 'Next_Tmin'. Next we will use the 'backelimination' to remove the variables that are not contributing much and we will fix our p-value at 0.95 (i.e remove the variables which have p-value > 0.95).

After the model is run, predict the Next_Tmax and Next_Tmin for 2017 and check them against the true values. Run this model for all the 25 stations and store all the summary and accuracy values in a dataframe. You can take the mean of temperature at a particular station and check the bias.

## Results
- Avg accuracy for Next_Tmax is: 79.4%
- Avg accuracy for Next_Tmin is: 84.3%













