import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix

class PreprocessingHelper:
    def __init__(self,dataframe=None):
        
        self.dataframe=dataframe

    def checkDf(self,dataframe=None,head=5,tail=5):
        """
        
        Prints shape, describe, info, head, tail, null values of dataframe.

        Parameters
        ------
            dataFrame: pandas.DataFrame, optional
                dataframe to be controlled.
                If not provided, the function uses the instance's DataFrame attribute.
            head: int, optional
                number of rows to select.
            tail: int, optional
                number of rows to select.

        Returns
        ------
            None: 
                The function prints shape, describe, info, head, tail, null values of dataframe.

        Examples
        ------
            import seaborn as sns
            df = sns.load_dataset("iris")
            checkDf(df)
        
        """
        if type(dataframe)!=pd.DataFrame:
            dataframe=self.dataframe
            if type(dataframe)!=pd.DataFrame:
                raise ValueError("Error: Missing parameter 'dataframe'. Please provide the required information.")
        
        print("##################### Shape #####################")
        print(dataframe.shape)
        print("##################### Describe #####################")
        print(dataframe.describe().T)
        print("##################### Info #####################")
        print(dataframe.info())
        print("##################### Head #####################")
        print(dataframe.head(head))
        print("##################### Tail #####################")
        print(dataframe.tail(tail))
        print("##################### NA #####################")
        print(dataframe.isnull().sum())

    def grabColNames(self,dataframe=None,car_threshold=20,cat_threshold=10):
        """

        Returns the names of categorical, numerical and categorical but cardinal variables in the data set.
        Note: Categorical variables with numerical appearance are also included in categorical variables.

        Parameters
        ------
            dataFrame: pandas.DataFrame, optional
                dataframe whose variable names are to be gotten
                If not provided, the function uses the instance's DataFrame attribute.
            car_threshold: int, optional
                threshold value for categorical but cardinal columns
            cat_threshold: int, optional
                threshold value for numerical but categorical columns
        
        Returns
        ------
            cat_cols: list
                categorical variable list
            cat_but_car: list
                categorical but cardinal variable list
            num_cols: list
                numerical variable list
        
        Examples
        ------
            import seaborn as sns
            df = sns.load_dataset("iris")
            cat_cols,cat_but_car,num_cols=grab_col_names(df)

        """
        if type(dataframe)!=pd.DataFrame:
            dataframe=self.dataframe
            if type(dataframe)!=pd.DataFrame:
                raise ValueError("Error: Missing parameter 'dataframe'. Please provide the required information.")

        cat_but_car=[col for col in dataframe.columns if (dataframe[col].dtype=="O") and (dataframe[col]).nunique()>car_threshold]
        cat_cols=[col for col in dataframe.columns if (dataframe[col].dtype=="O") and (col not in cat_but_car)]

        num_but_cat=[col for col in dataframe.columns if (dataframe[col].dtype !="O") and (dataframe[col].nunique()<cat_threshold)]
        num_cols=[col for col in dataframe.columns if (dataframe[col].dtype !="O") and (col not in num_but_cat)]

        cat_cols=cat_cols+num_but_cat
        print(f"Observations = {dataframe.shape[0]}\n")
        print(f"Variables = {dataframe.shape[1]}\n")
        print(f"catorical columns = {len(cat_cols)}\n")
        print(f"catorical but cardinal columns = {len(cat_but_car)}\n")
        print(f"numerical columns = {len(num_cols)}\n")
        print(f"numerical but categorical columns = {len(num_but_cat)}\n")

        return cat_cols,cat_but_car,num_cols
    
    def outlierThresholds(self,column_name,dataframe=None,q1=0.25,q3=0.75):
        """

        Calculates the lower and upper limits for identifying outliers in a dataset column.

        This function computes the lower and upper limits for identifying potential outliers
        in a specified column of a pandas DataFrame using the interquartile range (IQR) method.
        The IQR method defines potential outliers as data points that fall below the lower
        limit or above the upper limit.

        IQR=Q3-Q1
        low_limit=Q1-(1.5*IQR)
        up_limit=Q3+(1.5*IQR)

        Parameters
        ------
            column_name: str
                the name of the column to calculate outlier limits.
            dataFrame: pandas.DataFrame, optional
                dataframe containing the data.
                If not provided, the function uses the instance's DataFrame attribute.
            q1: float, optional 
                the quartile value for the lower boundary. Default is 0.25 (Q1).
            q3: float, optional 
                the quartile value for the upper boundary. Default is 0.75 (Q3).

        
        Returns
        ------
            low_limit: float
                The calculated lower limit for identifying outliers.
            up_limit: float 
                The calculated upper limit for identifying outliers.

        """
        if type(dataframe)!=pd.DataFrame:
            dataframe=self.dataframe
            if type(dataframe)!=pd.DataFrame:
                raise ValueError("Error: Missing parameter 'dataframe'. Please provide the required information.")
        
        quartile1=dataframe[column_name].quantile(q1)
        quartile3=dataframe[column_name].quantile(q3)
        interquantile_range=quartile3-quartile1
        low_limit=quartile1-(1.5*interquantile_range)
        up_limit=quartile3+(1.5*interquantile_range)
        return low_limit,up_limit

    def checkOutlier(self,column_name,dataframe=None,q1=0.25,q3=0.75):
        """

        Check for the presence of outliers in a specified column of a dataFrame.        

        Parameters
        ------
            column_name: str
                the name of the column to check outliers.
            dataFrame: pandas.DataFrame, optional
                dataframe containing the data.
                If not provided, the function uses the instance's DataFrame attribute.
            q1: float, optional 
                the quartile value for the lower boundary. Default is 0.25 (Q1).
            q3: float, optional 
                the quartile value for the upper boundary. Default is 0.75 (Q3).

        
        Returns
        ------
            has_outliers: bool 
                True if outliers are present in the specified column, False otherwise.
        
        """
        if type(dataframe)!=pd.DataFrame:
            dataframe=self.dataframe
            if type(dataframe)!=pd.DataFrame:
                raise ValueError("Error: Missing parameter 'dataframe'. Please provide the required information.")
        
        low_limit,up_limit=self.outlierThresholds(dataframe=dataframe,column_name=column_name,q1=q1,q3=q3)
        if dataframe[(dataframe[column_name]>up_limit) | (dataframe[column_name]<low_limit)].any(axis=None):
            return True
        else:
            return False

    def replaceWithThresholds(self,column_name,dataframe=None,q1=0.25,q3=0.75):
        """

        Replaces outliers in a dataset column with lower and upper limits.

        Parameters
        ------
            column_name: str
                the name of the column to replace with outlier limits.
            dataFrame: pandas.DataFrame, optional
                dataframe containing the data.
                If not provided, the function uses the instance's DataFrame attribute.
            q1: float, optional 
                the quartile value for the lower boundary. Default is 0.25 (Q1).
            q3: float, optional 
                the quartile value for the upper boundary. Default is 0.75 (Q3).
        
        Returns
        ------
            None:
                The function replaces outliers in a dataset column with lower and upper limits.
        
        """
        if type(dataframe)!=pd.DataFrame:
            dataframe=self.dataframe
            if type(dataframe)!=pd.DataFrame:
                raise ValueError("Error: Missing parameter 'dataframe'. Please provide the required information.")
        
        low_limit,up_limit=self.outlierThresholds(dataframe=dataframe,column_name=column_name,q1=q1,q3=q3)
        dataframe.loc[(dataframe[column_name]>up_limit),column_name]=up_limit
        dataframe.loc[(dataframe[column_name]<low_limit),column_name]=low_limit

    def missingValuesTable(self,dataframe=None,return_missing_columns=False):
        """

        Generates a table of missing values for columns in a pandas DataFrame.

        This function calculates and displays the count and ratio of missing values for each
        column in the specified DataFrame or the instance's DataFrame attribute. It provides insights
        into the extent of missing data, helping with data quality assessment.

        Parameters
        ------
            dataFrame: pandas.DataFrame, optional
                The DataFrame containing the data to analyze.
                If not provided, the function uses the instance's DataFrame attribute.
            return_missing_columns: bool, optional 
                Whether to return a list of columns with missing values. Default is False.

        Returns
        ------
            None: 
                The function prints the missing values statistics table.
                If return_missing_columns is True:
                    missing_columns: list
                        A list of column names with missing values.

        """

        if type(dataframe)!=pd.DataFrame:
            dataframe=self.dataframe
            if type(dataframe)!=pd.DataFrame:
                raise ValueError("Error: Missing parameter 'dataframe'. Please provide the required information.")
        
        temp=pd.DataFrame()
        missing_columns=[]
        for idx in dataframe.isnull().sum().index:
            if dataframe.isnull().sum()[idx]!=0:
                missing_columns.append(idx)
                temp[idx]=[dataframe.isnull().sum()[idx],dataframe.isnull().sum()[idx]/len(dataframe)]
        temp=temp.set_axis(["Count","Ratio"],axis="index")
        temp=temp.T
        temp["Count"]=temp["Count"].astype(int)
        print(temp)
        if return_missing_columns:
            return missing_columns
    
    def rareAnalyser(self,target,cat_cols,dataframe=None):
        """

        Analyzes categorical columns for rare categories and their impact on a target variable.

        This function provides insights into categorical columns by displaying the count, ratio,
        and mean of a target variable for each category in the specified DataFrame or the instance's
        DataFrame attribute. It helps identify rare categories and their potential influence on the target.

        Parameters
        ------
            target: str
                The name of the target variable column.
            cat_cols: list
                A list of column names with categorical data to be analyzed.
            dataFrame: pandas.DataFrame, optional
                The DataFrame containing the data to analyze.
                If not provided, the function uses the instance's DataFrame attribute.

        Returns
        ------
            None: 
                The function prints information about each categorical column's categories,
                including count, ratio, and target variable mean.

        """
        if type(dataframe)!=pd.DataFrame:
            dataframe=self.dataframe
            if type(dataframe)!=pd.DataFrame:
                raise ValueError("Error: Missing parameter 'dataframe'. Please provide the required information.")
        
        for col in cat_cols:
            print(col, ":", len(dataframe[col].value_counts()))
            print(pd.DataFrame({"Count": dataframe[col].value_counts(),
                                "Ratio": dataframe[col].value_counts() / len(dataframe),
                                f"{target}_mean": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

    def rareEncoder(self,rare_perc,cat_cols,dataframe=None,inplace=True):
        """

        Encodes rare categories in specified categorical columns of a DataFrame.

        This function identifies and encodes rare categories in the specified categorical columns
        of the DataFrame. A category is considered rare if its occurrence ratio is below the given
        threshold (rare_perc). Rare categories are replaced with the label 'Rare' in the DataFrame.

        Parameters
        ------
            rare_perc: float
                The threshold below which a category is considered rare (0 < rare_perc < 1).
            cat_cols: list
                A list of column names with categorical data to be processed.
            dataFrame: pandas.DataFrame, optional
                The DataFrame containing the data to be encoded.
                If not provided, the function uses the instance's DataFrame attribute.
            inplace: bool,optional
                Whether to apply the encoding in-place or return a modified copy.
                Default is True.

        Returns
        ------
            If inplace is True:
                dataframe: pandas.DataFrame 
                    The DataFrame with rare encoding applied, updated in place.
            If inplace is False:
                new_df: pandas.DataFrame
                    A copy of the DataFrame with rare categories encoded as 'Rare'.

        """
        if type(dataframe)!=pd.DataFrame:
            dataframe=self.dataframe
            if type(dataframe)!=pd.DataFrame:
                raise ValueError("Error: Missing parameter 'dataframe'. Please provide the required information.")
        
        if inplace:
            rare_columns = [col for col in cat_cols if (dataframe[col].value_counts() / len(dataframe) < rare_perc).sum() > 1]

            for col in rare_columns:
                tmp = dataframe[col].value_counts() / len(dataframe)
                rare_labels = tmp[tmp < rare_perc].index
                dataframe[col] = np.where(dataframe[col].isin(rare_labels), col+"_"+"Rare", dataframe[col])
            
            self.dataframe=dataframe
            return dataframe
        
        else:
            new_df = dataframe.copy()
            rare_columns = [col for col in cat_cols if (new_df[col].value_counts() / len(new_df) < rare_perc).sum() > 1]

            for col in rare_columns:
                tmp = new_df[col].value_counts() / len(new_df)
                rare_labels = tmp[tmp < rare_perc].index
                new_df[col] = np.where(new_df[col].isin(rare_labels), col+"_"+"Rare", new_df[col])

            return new_df

    def catSummary(self,col_name,dataframe=None,plot=False):
        """

        Generates a summary of a categorical column in a pandas DataFrame.

        This function displays a summary of a categorical column's unique values, their counts,
        and their corresponding ratios in the DataFrame. Optionally, it can also create and display
        a count plot of the categorical values.

        Parameters
        ------
            col_name: str
                The name of the categorical column to be summarized.
            dataFrame: pandas.DataFrame, optional
                The DataFrame containing the data to summarize.
                If not provided, the function uses the instance's DataFrame attribute.
            plot: bool, optional
                Whether to create and display a count plot. Default is False.

        Returns
        ------
            None:
                The function prints the categorical column's value counts and ratios.

        """
        if type(dataframe)!=pd.DataFrame:
            dataframe=self.dataframe
            if type(dataframe)!=pd.DataFrame:
                raise ValueError("Error: Missing parameter 'dataframe'. Please provide the required information.")
        
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

    def numSummary(self,numerical_col,dataframe=None,plot=False):
        """

        Generates a summary of a numerical column in a pandas DataFrame.

        This function provides a summary of statistics for a numerical column in the DataFrame,
        including basic descriptive statistics and optional histogram plot.

        Parameters
        ------
            numerical_col: str
                The name of the numerical column to be summarized.
            dataFrame: pandas.DataFrame, optional
                The DataFrame containing the data to summarize.
                If not provided, the function uses the instance's DataFrame attribute.
            plot: bool, optional
                Whether to create and display a count plot. Default is False.

        Returns
        ------
            None:
                The function prints the numerical column's descriptive statistics.

        """
        if type(dataframe)!=pd.DataFrame:
            dataframe=self.dataframe
            if type(dataframe)!=pd.DataFrame:
                raise ValueError("Error: Missing parameter 'dataframe'. Please provide the required information.")
        
        quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

        print(dataframe[[numerical_col]].describe(quantiles).T, end="\n\n")

        if plot:
            dataframe[numerical_col].hist(bins=20)
            plt.xlabel(numerical_col)
            plt.title(numerical_col)
            plt.show(block=True)

    def targetSummaryWithCat(self,target,categorical_col,dataframe=None):
        """

        Generate a summary of the mean target variable for each category in a categorical column.

        This function calculates and displays the mean of a target variable for each category in
        the specified categorical column of the DataFrame. It provides insights into how the target
        variable varies across different categories.

        Parameters
        ------
            target: str
                The name of the target variable column.
            categorical_col: str
                The name of the categorical column for which to calculate the summary.
            dataFrame: pandas.DataFrame, optional
                The DataFrame containing the data to summarize.
                If not provided, the function uses the instance's DataFrame attribute.

        Returns
        ------
            None:
                The function prints the mean target variable for each category.

        """
        if type(dataframe)!=pd.DataFrame:
            dataframe=self.dataframe
            if type(dataframe)!=pd.DataFrame:
                raise ValueError("Error: Missing parameter 'dataframe'. Please provide the required information.")
        
        if target!=categorical_col:
            print(pd.DataFrame({f"{target}_Mean": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")
            print("##########################################\n")

    def highCorrelatedCols(self,dataframe=None,plot=False,corr_th=0.90):
        """

        Identifies and optionally visualizes columns with high correlation in a pandas DataFrame.

        This function calculates the correlation matrix for the DataFrame and identifies columns
        with an absolute correlation coefficient greater than or equal to the specified threshold
        (corr_th). It can also create and display a heatmap of the correlation matrix for visual
        inspection.

        Parameters
        ------
            dataFrame: pandas.DataFrame, optional
                The DataFrame containing the data.
                If not provided, the function uses the instance's DataFrame attribute.
            plot: bool, optional
                Whether to create and display a heatmap. Default is False.
            corr_th: float, optional
                The correlation threshold for identifying high-correlation columns.
                Columns with an absolute correlation coefficient greater than or equal to this threshold
                will be considered highly correlated. Default is 0.90.

        Returns
        ------
            drop_list: list
                A list of column names with high correlation that can be considered for
                dropping or further analysis.

        """
        if type(dataframe)!=pd.DataFrame:
            dataframe=self.dataframe
            if type(dataframe)!=pd.DataFrame:
                raise ValueError("Error: Missing parameter 'dataframe'. Please provide the required information.")
        
        corr = dataframe.corr()
        cor_matrix = corr.abs()
        upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
        drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
        if plot:
            sns.set(rc={'figure.figsize': (15, 15)})
            sns.heatmap(corr, cmap="RdBu", annot=True)
            plt.show(block=True)
        return drop_list

    def plotConfusionMatrix(self,y,y_pred):
        """

        Plots a confusion matrix and displays accuracy score for a classification model.

        This function generates a confusion matrix based on the actual target values (y) and the
        predicted values (y_pred) from a classification model. It also calculates and displays
        the accuracy score of the model.

        Parameters
        ------
            y: array
                The array of actual target values.
            y_pred: array
                The array of predicted values from a classification model.

        Returns
        ------
            None:
                The function displays a heatmap of the confusion matrix and the accuracy score.

        """
        acc = round(accuracy_score(y,y_pred),2)
        cm=confusion_matrix(y,y_pred)
        sns.heatmap(cm,annot=True,fmt=".0f")
        plt.xlabel("y_pred")
        plt.ylabel("y")
        plt.title(f"Accuracy Score: {acc}")
        plt.show()

    def plotImportance(self,model,features,num=None,save=False):
        """

        Plots feature importances from a trained machine learning model.

        This function generates a bar plot of feature importances based on a trained machine learning model.
        It can display the specified number of top important features and optionally save the plot as an image.

        Parameters
        ------
            model: 
                The trained machine learning model with a `feature_importances_` attribute.
            features: pandas.DataFrame
                The DataFrame containing the features used for training.
            num: int, optional
                The number of top important features to display. Default is None, which displays all features.
            save: bool, optional
                Whether to save the plot as an image. Default is False.

        Returns
        ------
            None:
                The function displays a bar plot of feature importances.

        """
        if num==None:
            num=len(features)
        try:
            feature_importance=pd.DataFrame({
                "Value":model.feature_importances_,
                "Feature":features.columns
            })
        except:
            feature_importance=pd.DataFrame({
                "Value":abs(model.coef_[0]),
                "Feature":features.columns
            })
        plt.figure(figsize=(10,10))
        sns.set(font_scale=1)
        sns.barplot(x="Value",y="Feature",data=feature_importance.sort_values(by="Value",ascending=False)[0:num])
        plt.title("Features")
        plt.tight_layout()
        plt.show()
        if save:
            plt.savefig("importances.png")

    def missingValuesFiller(self,missing_columns=None,dataframe=None):
        """

        Fill missing values in specified columns of a pandas DataFrame.

        This function fills missing values in the specified columns of the DataFrame using either the
        median (for numerical columns) or mode (for categorical columns) of corresponding groups defined
        by non-missing values in other columns.

        Parameters
        ------
            missing_columns: list, optional
                A list of column names to fill missing values in. If not provided,
                the function uses the result from `missingValuesTable` to determine the columns with missing values.
            dataframe: pandas.DataFrame, optional
                The DataFrame containing the data to be filled.
                If not provided, the function uses the instance's dataframe attribute.

        Returns
        ------
            None:
                The function fills missing values in the DataFrame in place.

        """
        if type(dataframe)!=pd.DataFrame:
            dataframe=self.dataframe
            if type(dataframe)!=pd.DataFrame:
                raise ValueError("Error: Missing parameter 'dataframe'. Please provide the required information.")
        
        if missing_columns==None:
            missing_columns=self.missingValuesTable(return_missing_columns=True)
        
        temp_list=[col for col in dataframe.columns if (col not in missing_columns)]
        while dataframe.isnull().sum().any() and temp_list!=[]:
            for col in missing_columns:
                if dataframe[col].dtype=="O":
                    mode_selection = lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
                    dataframe[col] = dataframe[col].fillna(dataframe.groupby(temp_list)[col].transform(mode_selection))
                else:  
                    dataframe[col]=dataframe[col].fillna(dataframe.groupby(temp_list)[col].transform("median"))
    
            temp_list.pop(0)
    
    def oneHotEncoder(self,cat_cols,dataframe=None,drop_first=True,inplace=True):
        """

        Apply one-hot encoding to specified categorical columns of a pandas DataFrame.

        This function performs one-hot encoding on the specified categorical columns of the DataFrame.
        It creates binary columns for each category in the specified columns, where a value of 1
        indicates the presence of the category, and 0 indicates its absence. The original categorical
        columns can be dropped if 'drop_first' is set to True.

        Parameters
        ------
            rare_perc: float
                The threshold below which a category is considered rare (0 < rare_perc < 1).
            cat_cols: list
                A list of column names with categorical data to be processed.
            dataFrame: pandas.DataFrame, optional
                The DataFrame containing the data to be encoded.
                If not provided, the function uses the instance's DataFrame attribute.
            drop_first: bool, optional 
                Whether to drop the first category column in each categorical
                column. Default is True.
            inplace: bool,optional
                Whether to apply the encoding in-place or return a modified copy.
                Default is True.

        Returns
        ------
            If inplace is True:
                dataframe: pandas.DataFrame 
                    The DataFrame with one-hot encoding applied, updated in place.
            If inplace is False:
                new_df: pandas.DataFrame 
                    A modified copy of the DataFrame with one-hot encoding applied.

        """
        if type(dataframe)!=pd.DataFrame:
            dataframe=self.dataframe
            if type(dataframe)!=pd.DataFrame:
                raise ValueError("Error: Missing parameter 'dataframe'. Please provide the required information.")
        
        if inplace:
            for col in cat_cols:
                dataframe=pd.get_dummies(dataframe,columns=[col],drop_first=drop_first)
            
            self.dataframe=dataframe
            return dataframe
        
        else:
            new_df = dataframe.copy()
            for col in cat_cols:
                new_df=pd.get_dummies(new_df,columns=[col],drop_first=drop_first)

            return new_df
        
