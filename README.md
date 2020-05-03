# House-Sales-in-King-County-USA
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.preprocessing import StandardScaler,PolynomialFeatures\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "file_name='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv'\n",
    "df=pd.read_csv(file_name)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>id</th>\n",
       "      <th>date</th>\n",
       "      <th>price</th>\n",
       "      <th>bedrooms</th>\n",
       "      <th>bathrooms</th>\n",
       "      <th>sqft_living</th>\n",
       "      <th>sqft_lot</th>\n",
       "      <th>floors</th>\n",
       "      <th>waterfront</th>\n",
       "      <th>...</th>\n",
       "      <th>grade</th>\n",
       "      <th>sqft_above</th>\n",
       "      <th>sqft_basement</th>\n",
       "      <th>yr_built</th>\n",
       "      <th>yr_renovated</th>\n",
       "      <th>zipcode</th>\n",
       "      <th>lat</th>\n",
       "      <th>long</th>\n",
       "      <th>sqft_living15</th>\n",
       "      <th>sqft_lot15</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>7129300520</td>\n",
       "      <td>20141013T000000</td>\n",
       "      <td>221900.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.00</td>\n",
       "      <td>1180</td>\n",
       "      <td>5650</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>7</td>\n",
       "      <td>1180</td>\n",
       "      <td>0</td>\n",
       "      <td>1955</td>\n",
       "      <td>0</td>\n",
       "      <td>98178</td>\n",
       "      <td>47.5112</td>\n",
       "      <td>-122.257</td>\n",
       "      <td>1340</td>\n",
       "      <td>5650</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>6414100192</td>\n",
       "      <td>20141209T000000</td>\n",
       "      <td>538000.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>2.25</td>\n",
       "      <td>2570</td>\n",
       "      <td>7242</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>7</td>\n",
       "      <td>2170</td>\n",
       "      <td>400</td>\n",
       "      <td>1951</td>\n",
       "      <td>1991</td>\n",
       "      <td>98125</td>\n",
       "      <td>47.7210</td>\n",
       "      <td>-122.319</td>\n",
       "      <td>1690</td>\n",
       "      <td>7639</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>5631500400</td>\n",
       "      <td>20150225T000000</td>\n",
       "      <td>180000.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1.00</td>\n",
       "      <td>770</td>\n",
       "      <td>10000</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>6</td>\n",
       "      <td>770</td>\n",
       "      <td>0</td>\n",
       "      <td>1933</td>\n",
       "      <td>0</td>\n",
       "      <td>98028</td>\n",
       "      <td>47.7379</td>\n",
       "      <td>-122.233</td>\n",
       "      <td>2720</td>\n",
       "      <td>8062</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>2487200875</td>\n",
       "      <td>20141209T000000</td>\n",
       "      <td>604000.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>3.00</td>\n",
       "      <td>1960</td>\n",
       "      <td>5000</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>7</td>\n",
       "      <td>1050</td>\n",
       "      <td>910</td>\n",
       "      <td>1965</td>\n",
       "      <td>0</td>\n",
       "      <td>98136</td>\n",
       "      <td>47.5208</td>\n",
       "      <td>-122.393</td>\n",
       "      <td>1360</td>\n",
       "      <td>5000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>1954400510</td>\n",
       "      <td>20150218T000000</td>\n",
       "      <td>510000.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>2.00</td>\n",
       "      <td>1680</td>\n",
       "      <td>8080</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>8</td>\n",
       "      <td>1680</td>\n",
       "      <td>0</td>\n",
       "      <td>1987</td>\n",
       "      <td>0</td>\n",
       "      <td>98074</td>\n",
       "      <td>47.6168</td>\n",
       "      <td>-122.045</td>\n",
       "      <td>1800</td>\n",
       "      <td>7503</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 22 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0          id             date     price  bedrooms  bathrooms  \\\n",
       "0           0  7129300520  20141013T000000  221900.0       3.0       1.00   \n",
       "1           1  6414100192  20141209T000000  538000.0       3.0       2.25   \n",
       "2           2  5631500400  20150225T000000  180000.0       2.0       1.00   \n",
       "3           3  2487200875  20141209T000000  604000.0       4.0       3.00   \n",
       "4           4  1954400510  20150218T000000  510000.0       3.0       2.00   \n",
       "\n",
       "   sqft_living  sqft_lot  floors  waterfront  ...  grade  sqft_above  \\\n",
       "0         1180      5650     1.0           0  ...      7        1180   \n",
       "1         2570      7242     2.0           0  ...      7        2170   \n",
       "2          770     10000     1.0           0  ...      6         770   \n",
       "3         1960      5000     1.0           0  ...      7        1050   \n",
       "4         1680      8080     1.0           0  ...      8        1680   \n",
       "\n",
       "   sqft_basement  yr_built  yr_renovated  zipcode      lat     long  \\\n",
       "0              0      1955             0    98178  47.5112 -122.257   \n",
       "1            400      1951          1991    98125  47.7210 -122.319   \n",
       "2              0      1933             0    98028  47.7379 -122.233   \n",
       "3            910      1965             0    98136  47.5208 -122.393   \n",
       "4              0      1987             0    98074  47.6168 -122.045   \n",
       "\n",
       "   sqft_living15  sqft_lot15  \n",
       "0           1340        5650  \n",
       "1           1690        7639  \n",
       "2           2720        8062  \n",
       "3           1360        5000  \n",
       "4           1800        7503  \n",
       "\n",
       "[5 rows x 22 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Unnamed: 0         int64\n",
      "id                 int64\n",
      "date              object\n",
      "price            float64\n",
      "bedrooms         float64\n",
      "bathrooms        float64\n",
      "sqft_living        int64\n",
      "sqft_lot           int64\n",
      "floors           float64\n",
      "waterfront         int64\n",
      "view               int64\n",
      "condition          int64\n",
      "grade              int64\n",
      "sqft_above         int64\n",
      "sqft_basement      int64\n",
      "yr_built           int64\n",
      "yr_renovated       int64\n",
      "zipcode            int64\n",
      "lat              float64\n",
      "long             float64\n",
      "sqft_living15      int64\n",
      "sqft_lot15         int64\n",
      "dtype: object\n"
     ]
    }
   ],
   "source": [
    "print(df.dtypes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>id</th>\n",
       "      <th>price</th>\n",
       "      <th>bedrooms</th>\n",
       "      <th>bathrooms</th>\n",
       "      <th>sqft_living</th>\n",
       "      <th>sqft_lot</th>\n",
       "      <th>floors</th>\n",
       "      <th>waterfront</th>\n",
       "      <th>view</th>\n",
       "      <th>...</th>\n",
       "      <th>grade</th>\n",
       "      <th>sqft_above</th>\n",
       "      <th>sqft_basement</th>\n",
       "      <th>yr_built</th>\n",
       "      <th>yr_renovated</th>\n",
       "      <th>zipcode</th>\n",
       "      <th>lat</th>\n",
       "      <th>long</th>\n",
       "      <th>sqft_living15</th>\n",
       "      <th>sqft_lot15</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>21613.00000</td>\n",
       "      <td>2.161300e+04</td>\n",
       "      <td>2.161300e+04</td>\n",
       "      <td>21600.000000</td>\n",
       "      <td>21603.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>2.161300e+04</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>10806.00000</td>\n",
       "      <td>4.580302e+09</td>\n",
       "      <td>5.400881e+05</td>\n",
       "      <td>3.372870</td>\n",
       "      <td>2.115736</td>\n",
       "      <td>2079.899736</td>\n",
       "      <td>1.510697e+04</td>\n",
       "      <td>1.494309</td>\n",
       "      <td>0.007542</td>\n",
       "      <td>0.234303</td>\n",
       "      <td>...</td>\n",
       "      <td>7.656873</td>\n",
       "      <td>1788.390691</td>\n",
       "      <td>291.509045</td>\n",
       "      <td>1971.005136</td>\n",
       "      <td>84.402258</td>\n",
       "      <td>98077.939805</td>\n",
       "      <td>47.560053</td>\n",
       "      <td>-122.213896</td>\n",
       "      <td>1986.552492</td>\n",
       "      <td>12768.455652</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>6239.28002</td>\n",
       "      <td>2.876566e+09</td>\n",
       "      <td>3.671272e+05</td>\n",
       "      <td>0.926657</td>\n",
       "      <td>0.768996</td>\n",
       "      <td>918.440897</td>\n",
       "      <td>4.142051e+04</td>\n",
       "      <td>0.539989</td>\n",
       "      <td>0.086517</td>\n",
       "      <td>0.766318</td>\n",
       "      <td>...</td>\n",
       "      <td>1.175459</td>\n",
       "      <td>828.090978</td>\n",
       "      <td>442.575043</td>\n",
       "      <td>29.373411</td>\n",
       "      <td>401.679240</td>\n",
       "      <td>53.505026</td>\n",
       "      <td>0.138564</td>\n",
       "      <td>0.140828</td>\n",
       "      <td>685.391304</td>\n",
       "      <td>27304.179631</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.00000</td>\n",
       "      <td>1.000102e+06</td>\n",
       "      <td>7.500000e+04</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>290.000000</td>\n",
       "      <td>5.200000e+02</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>290.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1900.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>98001.000000</td>\n",
       "      <td>47.155900</td>\n",
       "      <td>-122.519000</td>\n",
       "      <td>399.000000</td>\n",
       "      <td>651.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>5403.00000</td>\n",
       "      <td>2.123049e+09</td>\n",
       "      <td>3.219500e+05</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.750000</td>\n",
       "      <td>1427.000000</td>\n",
       "      <td>5.040000e+03</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>1190.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1951.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>98033.000000</td>\n",
       "      <td>47.471000</td>\n",
       "      <td>-122.328000</td>\n",
       "      <td>1490.000000</td>\n",
       "      <td>5100.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>10806.00000</td>\n",
       "      <td>3.904930e+09</td>\n",
       "      <td>4.500000e+05</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>2.250000</td>\n",
       "      <td>1910.000000</td>\n",
       "      <td>7.618000e+03</td>\n",
       "      <td>1.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>1560.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1975.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>98065.000000</td>\n",
       "      <td>47.571800</td>\n",
       "      <td>-122.230000</td>\n",
       "      <td>1840.000000</td>\n",
       "      <td>7620.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>16209.00000</td>\n",
       "      <td>7.308900e+09</td>\n",
       "      <td>6.450000e+05</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>2.500000</td>\n",
       "      <td>2550.000000</td>\n",
       "      <td>1.068800e+04</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>2210.000000</td>\n",
       "      <td>560.000000</td>\n",
       "      <td>1997.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>98118.000000</td>\n",
       "      <td>47.678000</td>\n",
       "      <td>-122.125000</td>\n",
       "      <td>2360.000000</td>\n",
       "      <td>10083.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>21612.00000</td>\n",
       "      <td>9.900000e+09</td>\n",
       "      <td>7.700000e+06</td>\n",
       "      <td>33.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>13540.000000</td>\n",
       "      <td>1.651359e+06</td>\n",
       "      <td>3.500000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>13.000000</td>\n",
       "      <td>9410.000000</td>\n",
       "      <td>4820.000000</td>\n",
       "      <td>2015.000000</td>\n",
       "      <td>2015.000000</td>\n",
       "      <td>98199.000000</td>\n",
       "      <td>47.777600</td>\n",
       "      <td>-121.315000</td>\n",
       "      <td>6210.000000</td>\n",
       "      <td>871200.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>8 rows × 21 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        Unnamed: 0            id         price      bedrooms     bathrooms  \\\n",
       "count  21613.00000  2.161300e+04  2.161300e+04  21600.000000  21603.000000   \n",
       "mean   10806.00000  4.580302e+09  5.400881e+05      3.372870      2.115736   \n",
       "std     6239.28002  2.876566e+09  3.671272e+05      0.926657      0.768996   \n",
       "min        0.00000  1.000102e+06  7.500000e+04      1.000000      0.500000   \n",
       "25%     5403.00000  2.123049e+09  3.219500e+05      3.000000      1.750000   \n",
       "50%    10806.00000  3.904930e+09  4.500000e+05      3.000000      2.250000   \n",
       "75%    16209.00000  7.308900e+09  6.450000e+05      4.000000      2.500000   \n",
       "max    21612.00000  9.900000e+09  7.700000e+06     33.000000      8.000000   \n",
       "\n",
       "        sqft_living      sqft_lot        floors    waterfront          view  \\\n",
       "count  21613.000000  2.161300e+04  21613.000000  21613.000000  21613.000000   \n",
       "mean    2079.899736  1.510697e+04      1.494309      0.007542      0.234303   \n",
       "std      918.440897  4.142051e+04      0.539989      0.086517      0.766318   \n",
       "min      290.000000  5.200000e+02      1.000000      0.000000      0.000000   \n",
       "25%     1427.000000  5.040000e+03      1.000000      0.000000      0.000000   \n",
       "50%     1910.000000  7.618000e+03      1.500000      0.000000      0.000000   \n",
       "75%     2550.000000  1.068800e+04      2.000000      0.000000      0.000000   \n",
       "max    13540.000000  1.651359e+06      3.500000      1.000000      4.000000   \n",
       "\n",
       "       ...         grade    sqft_above  sqft_basement      yr_built  \\\n",
       "count  ...  21613.000000  21613.000000   21613.000000  21613.000000   \n",
       "mean   ...      7.656873   1788.390691     291.509045   1971.005136   \n",
       "std    ...      1.175459    828.090978     442.575043     29.373411   \n",
       "min    ...      1.000000    290.000000       0.000000   1900.000000   \n",
       "25%    ...      7.000000   1190.000000       0.000000   1951.000000   \n",
       "50%    ...      7.000000   1560.000000       0.000000   1975.000000   \n",
       "75%    ...      8.000000   2210.000000     560.000000   1997.000000   \n",
       "max    ...     13.000000   9410.000000    4820.000000   2015.000000   \n",
       "\n",
       "       yr_renovated       zipcode           lat          long  sqft_living15  \\\n",
       "count  21613.000000  21613.000000  21613.000000  21613.000000   21613.000000   \n",
       "mean      84.402258  98077.939805     47.560053   -122.213896    1986.552492   \n",
       "std      401.679240     53.505026      0.138564      0.140828     685.391304   \n",
       "min        0.000000  98001.000000     47.155900   -122.519000     399.000000   \n",
       "25%        0.000000  98033.000000     47.471000   -122.328000    1490.000000   \n",
       "50%        0.000000  98065.000000     47.571800   -122.230000    1840.000000   \n",
       "75%        0.000000  98118.000000     47.678000   -122.125000    2360.000000   \n",
       "max     2015.000000  98199.000000     47.777600   -121.315000    6210.000000   \n",
       "\n",
       "          sqft_lot15  \n",
       "count   21613.000000  \n",
       "mean    12768.455652  \n",
       "std     27304.179631  \n",
       "min       651.000000  \n",
       "25%      5100.000000  \n",
       "50%      7620.000000  \n",
       "75%     10083.000000  \n",
       "max    871200.000000  \n",
       "\n",
       "[8 rows x 21 columns]"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>price</th>\n",
       "      <th>bedrooms</th>\n",
       "      <th>bathrooms</th>\n",
       "      <th>sqft_living</th>\n",
       "      <th>sqft_lot</th>\n",
       "      <th>floors</th>\n",
       "      <th>waterfront</th>\n",
       "      <th>view</th>\n",
       "      <th>condition</th>\n",
       "      <th>grade</th>\n",
       "      <th>sqft_above</th>\n",
       "      <th>sqft_basement</th>\n",
       "      <th>yr_built</th>\n",
       "      <th>yr_renovated</th>\n",
       "      <th>zipcode</th>\n",
       "      <th>lat</th>\n",
       "      <th>long</th>\n",
       "      <th>sqft_living15</th>\n",
       "      <th>sqft_lot15</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>2.161300e+04</td>\n",
       "      <td>21600.000000</td>\n",
       "      <td>21603.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>2.161300e+04</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "      <td>21613.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>5.400881e+05</td>\n",
       "      <td>3.372870</td>\n",
       "      <td>2.115736</td>\n",
       "      <td>2079.899736</td>\n",
       "      <td>1.510697e+04</td>\n",
       "      <td>1.494309</td>\n",
       "      <td>0.007542</td>\n",
       "      <td>0.234303</td>\n",
       "      <td>3.409430</td>\n",
       "      <td>7.656873</td>\n",
       "      <td>1788.390691</td>\n",
       "      <td>291.509045</td>\n",
       "      <td>1971.005136</td>\n",
       "      <td>84.402258</td>\n",
       "      <td>98077.939805</td>\n",
       "      <td>47.560053</td>\n",
       "      <td>-122.213896</td>\n",
       "      <td>1986.552492</td>\n",
       "      <td>12768.455652</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>3.671272e+05</td>\n",
       "      <td>0.926657</td>\n",
       "      <td>0.768996</td>\n",
       "      <td>918.440897</td>\n",
       "      <td>4.142051e+04</td>\n",
       "      <td>0.539989</td>\n",
       "      <td>0.086517</td>\n",
       "      <td>0.766318</td>\n",
       "      <td>0.650743</td>\n",
       "      <td>1.175459</td>\n",
       "      <td>828.090978</td>\n",
       "      <td>442.575043</td>\n",
       "      <td>29.373411</td>\n",
       "      <td>401.679240</td>\n",
       "      <td>53.505026</td>\n",
       "      <td>0.138564</td>\n",
       "      <td>0.140828</td>\n",
       "      <td>685.391304</td>\n",
       "      <td>27304.179631</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>7.500000e+04</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>290.000000</td>\n",
       "      <td>5.200000e+02</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>290.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1900.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>98001.000000</td>\n",
       "      <td>47.155900</td>\n",
       "      <td>-122.519000</td>\n",
       "      <td>399.000000</td>\n",
       "      <td>651.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>3.219500e+05</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.750000</td>\n",
       "      <td>1427.000000</td>\n",
       "      <td>5.040000e+03</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>1190.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1951.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>98033.000000</td>\n",
       "      <td>47.471000</td>\n",
       "      <td>-122.328000</td>\n",
       "      <td>1490.000000</td>\n",
       "      <td>5100.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>4.500000e+05</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>2.250000</td>\n",
       "      <td>1910.000000</td>\n",
       "      <td>7.618000e+03</td>\n",
       "      <td>1.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>1560.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1975.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>98065.000000</td>\n",
       "      <td>47.571800</td>\n",
       "      <td>-122.230000</td>\n",
       "      <td>1840.000000</td>\n",
       "      <td>7620.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>6.450000e+05</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>2.500000</td>\n",
       "      <td>2550.000000</td>\n",
       "      <td>1.068800e+04</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>2210.000000</td>\n",
       "      <td>560.000000</td>\n",
       "      <td>1997.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>98118.000000</td>\n",
       "      <td>47.678000</td>\n",
       "      <td>-122.125000</td>\n",
       "      <td>2360.000000</td>\n",
       "      <td>10083.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>7.700000e+06</td>\n",
       "      <td>33.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>13540.000000</td>\n",
       "      <td>1.651359e+06</td>\n",
       "      <td>3.500000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>13.000000</td>\n",
       "      <td>9410.000000</td>\n",
       "      <td>4820.000000</td>\n",
       "      <td>2015.000000</td>\n",
       "      <td>2015.000000</td>\n",
       "      <td>98199.000000</td>\n",
       "      <td>47.777600</td>\n",
       "      <td>-121.315000</td>\n",
       "      <td>6210.000000</td>\n",
       "      <td>871200.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              price      bedrooms     bathrooms   sqft_living      sqft_lot  \\\n",
       "count  2.161300e+04  21600.000000  21603.000000  21613.000000  2.161300e+04   \n",
       "mean   5.400881e+05      3.372870      2.115736   2079.899736  1.510697e+04   \n",
       "std    3.671272e+05      0.926657      0.768996    918.440897  4.142051e+04   \n",
       "min    7.500000e+04      1.000000      0.500000    290.000000  5.200000e+02   \n",
       "25%    3.219500e+05      3.000000      1.750000   1427.000000  5.040000e+03   \n",
       "50%    4.500000e+05      3.000000      2.250000   1910.000000  7.618000e+03   \n",
       "75%    6.450000e+05      4.000000      2.500000   2550.000000  1.068800e+04   \n",
       "max    7.700000e+06     33.000000      8.000000  13540.000000  1.651359e+06   \n",
       "\n",
       "             floors    waterfront          view     condition         grade  \\\n",
       "count  21613.000000  21613.000000  21613.000000  21613.000000  21613.000000   \n",
       "mean       1.494309      0.007542      0.234303      3.409430      7.656873   \n",
       "std        0.539989      0.086517      0.766318      0.650743      1.175459   \n",
       "min        1.000000      0.000000      0.000000      1.000000      1.000000   \n",
       "25%        1.000000      0.000000      0.000000      3.000000      7.000000   \n",
       "50%        1.500000      0.000000      0.000000      3.000000      7.000000   \n",
       "75%        2.000000      0.000000      0.000000      4.000000      8.000000   \n",
       "max        3.500000      1.000000      4.000000      5.000000     13.000000   \n",
       "\n",
       "         sqft_above  sqft_basement      yr_built  yr_renovated       zipcode  \\\n",
       "count  21613.000000   21613.000000  21613.000000  21613.000000  21613.000000   \n",
       "mean    1788.390691     291.509045   1971.005136     84.402258  98077.939805   \n",
       "std      828.090978     442.575043     29.373411    401.679240     53.505026   \n",
       "min      290.000000       0.000000   1900.000000      0.000000  98001.000000   \n",
       "25%     1190.000000       0.000000   1951.000000      0.000000  98033.000000   \n",
       "50%     1560.000000       0.000000   1975.000000      0.000000  98065.000000   \n",
       "75%     2210.000000     560.000000   1997.000000      0.000000  98118.000000   \n",
       "max     9410.000000    4820.000000   2015.000000   2015.000000  98199.000000   \n",
       "\n",
       "                lat          long  sqft_living15     sqft_lot15  \n",
       "count  21613.000000  21613.000000   21613.000000   21613.000000  \n",
       "mean      47.560053   -122.213896    1986.552492   12768.455652  \n",
       "std        0.138564      0.140828     685.391304   27304.179631  \n",
       "min       47.155900   -122.519000     399.000000     651.000000  \n",
       "25%       47.471000   -122.328000    1490.000000    5100.000000  \n",
       "50%       47.571800   -122.230000    1840.000000    7620.000000  \n",
       "75%       47.678000   -122.125000    2360.000000   10083.000000  \n",
       "max       47.777600   -121.315000    6210.000000  871200.000000  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.drop(['id', 'Unnamed: 0'], axis=1, inplace=True)\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "number of NaN values for the column bedrooms : 13\n",
      "number of NaN values for the column bathrooms : 10\n"
     ]
    }
   ],
   "source": [
    "print(\"number of NaN values for the column bedrooms :\", df['bedrooms'].isnull().sum())\n",
    "print(\"number of NaN values for the column bathrooms :\", df['bathrooms'].isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "mean=df['bedrooms'].mean()\n",
    "df['bedrooms'].replace(np.nan,mean, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "mean=df['bathrooms'].mean()\n",
    "df['bathrooms'].replace(np.nan,mean, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "number of NaN values for the column bedrooms : 0\n",
      "number of NaN values for the column bathrooms : 0\n"
     ]
    }
   ],
   "source": [
    "print(\"number of NaN values for the column bedrooms :\", df['bedrooms'].isnull().sum())\n",
    "print(\"number of NaN values for the column bathrooms :\", df['bathrooms'].isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>floors</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1.0</th>\n",
       "      <td>10680</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2.0</th>\n",
       "      <td>8241</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1.5</th>\n",
       "      <td>1910</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3.0</th>\n",
       "      <td>613</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2.5</th>\n",
       "      <td>161</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3.5</th>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     floors\n",
       "1.0   10680\n",
       "2.0    8241\n",
       "1.5    1910\n",
       "3.0     613\n",
       "2.5     161\n",
       "3.5       8"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['floors'].value_counts().to_frame()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x7f3cad5a4400>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(x='waterfront', y='price', data=df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x7f3cad522588>"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.regplot(x='sqft_above', y='price', data=df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "zipcode         -0.053203\n",
       "long             0.021626\n",
       "condition        0.036362\n",
       "yr_built         0.054012\n",
       "sqft_lot15       0.082447\n",
       "sqft_lot         0.089661\n",
       "yr_renovated     0.126434\n",
       "floors           0.256794\n",
       "waterfront       0.266369\n",
       "lat              0.307003\n",
       "bedrooms         0.308797\n",
       "sqft_basement    0.323816\n",
       "view             0.397293\n",
       "bathrooms        0.525738\n",
       "sqft_living15    0.585379\n",
       "sqft_above       0.605567\n",
       "grade            0.667434\n",
       "sqft_living      0.702035\n",
       "price            1.000000\n",
       "Name: price, dtype: float64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.corr()['price'].sort_values()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "from sklearn.linear_model import LinearRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.00046769430149007363"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = df[['long']]\n",
    "Y = df['price']\n",
    "lm = LinearRegression()\n",
    "lm\n",
    "lm.fit(X,Y)\n",
    "lm.score(X, Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.49285321790379316"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = df[['sqft_living']]\n",
    "Y = df['price']\n",
    "lm = LinearRegression()\n",
    "lm.fit(X, Y)\n",
    "lm.score(X, Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "features =[\"floors\", \"waterfront\",\"lat\" ,\"bedrooms\" ,\"sqft_basement\" ,\"view\" ,\"bathrooms\",\"sqft_living15\",\"sqft_above\",\"grade\",\"sqft_living\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.6576527411217378"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = df[features]\n",
    "Y= df['price']\n",
    "lm = LinearRegression()\n",
    "lm.fit(X, Y)\n",
    "lm.score(X, Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,\n",
       "         normalize=False)"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'scale'\n",
    "\n",
    "'polynomial'\n",
    "\n",
    "'model'\n",
    "\n",
    "'The second element in the tuple contains the model constructor'\n",
    "\n",
    "StandardScaler()\n",
    "\n",
    "PolynomialFeatures(include_bias=False)\n",
    "\n",
    "LinearRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pipe=Pipeline(Input)\n",
    "pipe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/sklearn/preprocessing/data.py:625: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.\n",
      "  return self.partial_fit(X, y)\n",
      "/home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/sklearn/base.py:465: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.\n",
      "  return self.fit(X, y, **fit_params).transform(X)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Pipeline(memory=None,\n",
       "     steps=[('scale', StandardScaler(copy=True, with_mean=True, with_std=True)), ('polynomial', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)), ('model', LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,\n",
       "         normalize=False))])"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pipe.fit(X,Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/sklearn/pipeline.py:511: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.\n",
      "  Xt = transform.transform(Xt)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0.7513406368483374"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pipe.score(X,Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.model_selection import train_test_split\n",
    "print(\"done\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "number of test samples : 3242\n",
      "number of training samples: 18371\n"
     ]
    }
   ],
   "source": [
    "features =[\"floors\", \"waterfront\",\"lat\" ,\"bedrooms\" ,\"sqft_basement\" ,\"view\" ,\"bathrooms\",\"sqft_living15\",\"sqft_above\",\"grade\",\"sqft_living\"]    \n",
    "X = df[features ]\n",
    "Y = df['price']\n",
    "\n",
    "x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)\n",
    "\n",
    "\n",
    "print(\"number of test samples :\", x_test.shape[0])\n",
    "print(\"number of training samples:\",x_train.shape[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import Ridge"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.6478759163939115"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "RidgeModel = Ridge(alpha = 0.1)\n",
    "RidgeModel.fit(x_train, y_train)\n",
    "RidgeModel.score(x_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7002744288456159"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.preprocessing import PolynomialFeatures\n",
    "from sklearn.linear_model import Ridge\n",
    "pr = PolynomialFeatures(degree=2)\n",
    "x_train_pr = pr.fit_transform(x_train)\n",
    "x_test_pr = pr.fit_transform(x_test)\n",
    "poly = Ridge(alpha=0.1)\n",
    "poly.fit(x_train_pr, y_train)\n",
    "poly.score(x_test_pr, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python",
   "language": "python",
   "name": "conda-env-python-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
