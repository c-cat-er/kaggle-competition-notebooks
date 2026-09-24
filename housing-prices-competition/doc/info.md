# House Prices Advanced Regression

## 1. File Descriptions & Path Context

- https://drive.google.com/drive/folders/1vz_Zee98m5c38gGgaped17H5GM0q60PA?usp=drive_link
- **`train.csv`**: The training set.
- **`test.csv`**: The test set.
- **`data_description.txt`**: Full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here.
- **`sample_submission.csv`**: A benchmark submission from a linear regression on year and month of sale, lot square footage, and number of bedrooms.

## 2. Target Variable & Evaluation Metric

- **Target Variable**: `SalePrice` - The property's sale price in dollars. This is the target variable that you're trying to predict.
- **Metric**: Root Mean Squared Error (RMSE) between the logarithm of the predicted value and the logarithm of the actual sale price.
- **Required Pipeline Math**:
    - **Pre-training (Log-transform Target)**:
        ```python
        y_train = np.log1p(df['SalePrice'])
        ```
    - **Post-inference (Inverse-transform Predictions)**:
        ```python
        predictions = np.expm1(pred)
        ```

## 3. Data Preprocessing & Feature Engineering Requirements

- **Data Characteristics**: 79 explanatory variables (features) total, including continuous and categorical data.
- **Missing Value Imputation**: Dataset contains missing values. Code generation must include preprocessing steps for either imputation or deletion.
- **Feature Engineering**: Requires simplifying and merging existing features, and performing feature transformations (such as univariate/multivariate transformations).

## 4. Full Data Fields Definition (79 Features + 1 Target)

- **Target Variable**:
    - `SalePrice`: The property's sale price in dollars.
- **Property Attributes**:
    - `MSSubClass`: The building class.
    - `MSZoning`: The general zoning classification.
    - `LotFrontage`: Linear feet of street connected to property.
    - `LotArea`: Lot size in square feet.
    - `Street`: Type of road access.
    - `Alley`: Type of alley access.
    - `LotShape`: General shape of property.
    - `LandContour`: Flatness of the property.
    - `Utilities`: Type of utilities available.
    - `LotConfig`: Lot configuration.
    - `LandSlope`: Slope of property.
    - `Neighborhood`: Physical locations within Ames city limits.
    - `Condition1`: Proximity to main road or railroad.
    - `Condition2`: Proximity to main road or railroad (if a second is present).
    - `BldgType`: Type of dwelling.
    - `HouseStyle`: Style of dwelling.
- **Quality & History**:
    - `OverallQual`: Overall material and finish quality.
    - `OverallCond`: Overall condition rating.
    - `YearBuilt`: Original construction date.
    - `YearRemodAdd`: Remodel date.
- **Exterior Features**:
    - `RoofStyle`: Type of roof.
    - `RoofMatl`: Roof material.
    - `Exterior1st`: Exterior covering on house.
    - `Exterior2nd`: Exterior covering on house (if more than one material).
    - `MasVnrType`: Masonry veneer type.
    - `MasVnrArea`: Masonry veneer area in square feet.
    - `ExterQual`: Exterior material quality.
    - `ExterCond`: Present condition of the material on the exterior.
    - `Foundation`: Type of foundation.
- **Basement Attributes**:
    - `BsmtQual`: Height of the basement.
    - `BsmtCond`: General condition of the basement.
    - `BsmtExposure`: Walkout or garden level basement walls.
    - `BsmtFinType1`: Quality of basement finished area.
    - `BsmtFinSF1`: Type 1 finished square feet.
    - `BsmtFinType2`: Quality of second finished area (if present).
    - `BsmtFinSF2`: Type 2 finished square feet.
    - `BsmtUnfSF`: Unfinished square feet of basement area.
    - `TotalBsmtSF`: Total square feet of basement area.
- **Utilities & Internal Space**:
    - `Heating`: Type of heating.
    - `HeatingQC`: Heating quality and condition.
    - `CentralAir`: Central air conditioning.
    - `Electrical`: Electrical system.
    - `1stFlrSF`: First Floor square feet.
    - `2ndFlrSF`: Second floor square feet.
    - `LowQualFinSF`: Low quality finished square feet (all floors).
    - `GrLivArea`: Above grade (ground) living area square feet.
    - `BsmtFullBath`: Basement full bathrooms.
    - `BsmtHalfBath`: Basement half bathrooms.
    - `FullBath`: Full bathrooms above grade.
    - `HalfBath`: Half baths above grade.
    - `Bedroom`: Number of bedrooms above basement level.
    - `Kitchen`: Number of kitchens.
    - `KitchenQual`: Kitchen quality.
    - `TotRmsAbvGrd`: Total rooms above grade (does not include bathrooms).
    - `Functional`: Home functionality rating.
    - `Fireplaces`: Number of fireplaces.
    - `FireplaceQu`: Fireplace quality.
- **Garage Attributes**:
    - `GarageType`: Garage location.
    - `GarageYrBlt`: Year garage was built.
    - `GarageFinish`: Interior finish of the garage.
    - `GarageCars`: Size of garage in car capacity.
    - `GarageArea`: Size of garage in square feet.
    - `GarageQual`: Garage quality.
    - `GarageCond`: Garage condition.
- **Outdoor & Miscellaneous**:
    - `PavedDrive`: Paved driveway.
    - `WoodDeckSF`: Wood deck area in square feet.
    - `OpenPorchSF`: Open porch area in square feet.
    - `EnclosedPorch`: Enclosed porch area in square feet.
    - `3SsnPorch`: Three season porch area in square feet.
    - `ScreenPorch`: Screen porch area in square feet.
    - `PoolArea`: Pool area in square feet.
    - `PoolQC`: Pool quality.
    - `Fence`: Fence quality.
    - `MiscFeature`: Miscellaneous feature not covered in other categories.
    - `MiscVal`: \$Value of miscellaneous feature.
- **Transaction Metadata**:
    - `MoSold`: Month Sold.
    - `YrSold`: Year Sold.
    - `SaleType`: Type of sale.
    - `SaleCondition`: Condition of sale.

## 5. Recommended Algorithm Architectures

- **Linear Models & Regularization**: Ridge, LASSO (to avoid multicollinearity), and ElasticNet.
- **Tree-based & Boosting**: Random Forest, GBM, XGBoost, and ranger.
- **Deep Learning & Ensembling**: Neural Networks built with Keras, or multi-model integration via caret/Stacking methods.

## 6. Submission Export Constraints

- **File Format**: Comma-separated values (CSV).
- **Header Line Requirement**: Must be `Id,SalePrice` as the first line.
- **Data Row Schema**:
    ```csv
    Id,SalePrice
    1461,169000.1
    ```
