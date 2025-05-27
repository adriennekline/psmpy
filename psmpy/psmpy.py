import warnings
from .functions import cohenD
import numpy as np
from scipy.special import logit, expit
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import math
import pandas.api.types as ptypes
import seaborn as sns
sns.set(rc={'figure.figsize': (10, 8)}, font_scale=1.3)


class PsmPy:
    """
    Matcher Class -- Match data for an observational study.
    Parameters
    ----------
    data : pd.DataFrame
        Data representing the treated group
    treatment : str
        Column representing the intervention (binary)
    indx : str
        Name of patient index column
    exclude : list (optional)
        List of variables to ignore in regression/matching.
    target : str (optional)
        Outcome variable of interest, will ignore in regression/matching
    ----------
    """

    def __init__(self, data, treatment, indx, exclude=[], target='outcome', seed=42):
        # variables generated during matching
        # assign unique indices to test and control
        self.data = data.dropna(axis=0, how="all")  # drop all NAN rows
        self.data = data.dropna(axis=1, how="all")  # drop all NAN col
        self.data[treatment] = self.data[treatment].astype(
            int)  # need binary 0, 1
        self.control_color = "#1F77B4"
        self.treatment_color = "#FF7F0E"
        self.treatment = treatment
        self.target = target
        self.indx = indx
        self.exclude = exclude + [self.treatment] + [self.indx]
        self.drop = exclude
        self.xvars = [i for i in self.data.columns if i not in self.exclude]
        self.keep_cols = [self.indx] + self.xvars
        self.data = self.data.drop(labels=self.drop, axis=1)
        self.model_accuracy = []
        self.dataIDindx = self.data.set_index(indx)
        assert all(ptypes.is_numeric_dtype(
            self.dataIDindx[xvar]) for xvar in self.xvars), "Only numeric dtypes allowed"
        self.treatmentdf = self.dataIDindx[self.dataIDindx[treatment] == 1]
        self.controldf = self.dataIDindx[self.dataIDindx[treatment] == 0]
        self.treatmentn = len(self.treatmentdf)
        self.controln = len(self.controldf)
        self.seed = seed

    def logistic_ps(self, balance=True):
        """ 
        Fits logistic regression model(s) used for generating propensity scores
        Parameters
        ----------
        balance : bool
            Should balanced datasets be used?
            (n_control == n_test is default when balance = True)
        Returns
        predicted_data : pd.DataFrame
            DataFrame with propensity scores and logit propensity scores
        Returns
        matched_ids : pd.DataFrame
            DataFrame with propensity or logit scores
        -------
        """
        if self.treatmentn < self.controln:
            minority, majority = self.treatmentdf, self.controldf
        elif self.treatmentn > self.controln:
            minority, majority = self.controldf, self.treatmentdf
        else:
            minority, majority = self.controldf, self.treatmentdf
        # if user wishes cases to be balanced:
        if balance == True:
            def chunker(seq, size):
                return (seq[pos:pos + size] for pos in range(0, len(seq), size))

            even_folds = math.floor(len(majority)/minority.shape[0])
            majority_trunc_len = even_folds*minority.shape[0]
            majority_len_diff = majority.shape[0] - majority_trunc_len
            majority_trunc = majority[0:majority_trunc_len]

            appended_data = []
            for i in chunker(majority_trunc, minority.shape[0]):
                joint_df = pd.concat([i, minority])
                treatment = joint_df[self.treatment]
                df_cleaned = joint_df.drop([self.treatment], axis=1)
                logistic = LogisticRegression(solver='liblinear')
                logistic.fit(df_cleaned, treatment)
                pscore = logistic.predict_proba(df_cleaned)[:, 1]
                df_cleaned['propensity_score'] = pscore
                # df_cleaned['propensity_logit'] = np.array(
                # logit(xi) for xi in pscore)
                df_cleaned['propensity_logit'] = df_cleaned['propensity_score'].apply(
                    lambda p: np.log(p/(1-p)) if p < 0.9999 else np.log(p/(0.00001)))
                appended_data.append(df_cleaned)
            # if some of majority class leftover after the folding for training:
            if majority_len_diff != 0:
                majority_leftover = majority[majority_trunc_len:]
                len_major_leftover = len(majority_leftover)
                if len_major_leftover <= 20:
                    need_2_sample_more_majorclass = 20 - len_major_leftover
                    # select the remaining ones from major class:
                    majority_leftover1 = majority[majority_trunc_len:]
                    # select the ones that have already been processed:
                    majority_already_folded = majority[:majority_trunc_len]
                    # self.majority_already_folded = majority_already_folded
                    # sample from the ones already folded over above to get major class of 20 for last fold:
                    majority_leftover2 = majority_already_folded.sample(
                        n=need_2_sample_more_majorclass, random_state=self.seed)
                    # add the leftover to the additional that need to be sampled to get to 20
                    majority_leftover_all = pd.concat(
                        [majority_leftover1, majority_leftover2])
                    # sample a macthing 20 from the minor class
                    minority_sample = minority.sample(
                        n=20, random_state=self.seed)
                    joint_df = pd.concat(
                        [majority_leftover_all, minority_sample])
                    treatment = joint_df[self.treatment]
                    df_cleaned = joint_df.drop([self.treatment], axis=1)
                    logistic = LogisticRegression(solver='liblinear')
                    logistic.fit(df_cleaned, treatment)
                    pscore = logistic.predict_proba(df_cleaned)[:, 1]
                    df_cleaned['propensity_score'] = pscore
                    df_cleaned['propensity_logit'] = df_cleaned['propensity_score'].apply(
                        lambda p: np.log(p/(1-p)) if p < 0.9999 else np.log(p/(0.00001)))
                    appended_data.append(df_cleaned)
                else:
                    majority_extra = majority[majority_trunc_len:]
                    minority_sample = minority.sample(
                        n=majority_len_diff, random_state=self.seed)
                    joint_df = pd.concat([majority_extra, minority_sample])
                    treatment = joint_df[self.treatment]
                    df_cleaned = joint_df.drop([self.treatment], axis=1)
                    logistic = LogisticRegression(solver='liblinear')
                    logistic.fit(df_cleaned, treatment)
                    pscore = logistic.predict_proba(df_cleaned)[:, 1]
                    df_cleaned['propensity_score'] = pscore
                    df_cleaned['propensity_logit'] = df_cleaned['propensity_score'].apply(
                        lambda p: np.log(p/(1-p)) if p < 0.9999 else np.log(p/(0.00001)))
                    appended_data.append(df_cleaned)
            else:
                pass
            predicted_data_repeated = pd.concat(appended_data)
            predicted_data_repeated_reset = predicted_data_repeated.reset_index()

            # pull repeated minority class out to average calculations for prop scores
            repeated_data = predicted_data_repeated_reset[predicted_data_repeated_reset.duplicated(
                subset=self.indx, keep=False)]
            unique_ids = repeated_data[self.indx].unique()

            mean_repeated = []
            for repeated_id in unique_ids:
                temp_repeat_df = predicted_data_repeated_reset[
                    predicted_data_repeated_reset[self.indx] == repeated_id]
                prop_mean = temp_repeat_df['propensity_score'].mean()
                prop_logit_mean = logit(prop_mean)
                short_entry = temp_repeat_df[0: 1]
                short_entry_rst = short_entry.reset_index(drop=True)
                short_entry_rst.at[0, 'propensity_score'] = prop_mean
                short_entry_rst.at[0, 'propensity_logit'] = prop_logit_mean
                mean_repeated.append(short_entry_rst)
            filtered_repeated = pd.concat(mean_repeated)

            # remove all duplicated minority class from folded df to be rejoined with the fixed values
            not_repeated_predicted = predicted_data_repeated_reset.drop_duplicates(
                subset=self.indx, keep=False)
            predicted_data_ps = pd.concat(
                [filtered_repeated, not_repeated_predicted]).reset_index(drop=True)

            # merge with treatment df
            treatment_dfonly = self.dataIDindx[[self.treatment]].reset_index()
            self.predicted_data = pd.merge(
                predicted_data_ps, treatment_dfonly, how='inner', on=self.indx)
            predicted_data_control = self.predicted_data[self.predicted_data[self.treatment] == 0]
            predicted_data_treatment = self.predicted_data[self.predicted_data[self.treatment] == 1]

            # return predicted_data
        # If user does not wish cases to be balanced
        else:
            joint_df = pd.concat([majority, minority])
            treatment = joint_df[self.treatment]
            df_cleaned = joint_df.drop([self.treatment], axis=1)
            logistic = LogisticRegression(solver='liblinear')
            logistic.fit(df_cleaned, treatment)
            pscore = logistic.predict_proba(df_cleaned)[:, 1]
            df_cleaned['propensity_score'] = pscore
            df_cleaned['propensity_logit'] = df_cleaned['propensity_score'].apply(
                lambda p: np.log(p/(1-p)) if p < 0.9999 else np.log(p/(0.00001)))
            predicted_data_reset = df_cleaned.reset_index()

            # merge with treatment df
            treatment_dfonly = self.dataIDindx[[self.treatment]].reset_index()
            self.predicted_data = pd.merge(
                predicted_data_reset, treatment_dfonly, how='inner', on=self.indx)
            predicted_data_control = self.predicted_data[self.predicted_data[self.treatment] == 0]
            predicted_data_treatment = self.predicted_data[self.predicted_data[self.treatment] == 1]
            # return predicted_data

    def knn_matched(self, matcher='propensity_logit', replacement=False, caliper=None, drop_unmatched=True):
        """
        knn_matched -- Match data using k-nn algorithm
        Parameters
        ----------
        matcher : str
           string that will used to match - propensity score or proppensity logit
        replacement : bool
           Want to match with or without replacement (default = False)
        caliper : float
           caliper multiplier for allowable matching
        Returns
        matched_ids : pd.DataFrame
            DataFrame with column with matched ID based on k-NN algorithm
        """
        matcher = matcher
        predicted_data_control = self.predicted_data[self.predicted_data[self.treatment] == 0]
        predicted_data_treatment = self.predicted_data[self.predicted_data[self.treatment] == 1]

        # if caliper_multip is not None:
        # caliper = np.std(predicted_data_control[matcher]) * caliper_multip
        # else:
        # pass

        if len(predicted_data_treatment) < len(predicted_data_control):
            min_pred, major_pred = predicted_data_treatment, predicted_data_control
            major_pred_rstindx = major_pred.reset_index(drop=True)
            minor_pred_rstindx = min_pred.reset_index(drop=True)
        elif len(predicted_data_treatment) > len(predicted_data_control):
            min_pred, major_pred = predicted_data_control, predicted_data_treatment
            major_pred_rstindx = major_pred.reset_index(drop=True)
            minor_pred_rstindx = min_pred.reset_index(drop=True)
        else:
            min_pred, major_pred = predicted_data_control, predicted_data_treatment
            major_pred_rstindx = major_pred.reset_index(drop=True)
            minor_pred_rstindx = min_pred.reset_index(drop=True)

        # need to fit KNN with larger class
        knn = NearestNeighbors(n_neighbors=len(major_pred_rstindx), p=2)
        knn.fit(major_pred_rstindx[[matcher]].to_numpy())
        distances, indexes = knn.kneighbors(
            minor_pred_rstindx[[matcher]].to_numpy(), n_neighbors=len(major_pred_rstindx))
#         self.distances = distances
#         self.indexes = indexes

        def condition_caliper(x, caliper):
            return x <= caliper

        # remove elements outside of radius:
        if caliper is not None:
            indices_for_match = []
            elements_to_remove = []
            # loop through both distance and indexes from knn simultaneously:
            for dist, row in zip(distances[:, :], indexes[:, :]):
                # convert dist and row_ids to lists:
                dist = np.ndarray.tolist(dist)
                row = np.ndarray.tolist(row)
                # finds indices to include from distances based on caliper:
                dist_indices = [idx for idx, element in enumerate(
                    dist) if condition_caliper(element, caliper)]
                # clean up the ids from the distance exclusion:
                row_clean = [row[index] for index in dist_indices]
                # check to see if replacement is False:
                if replacement == False:
                    if len(elements_to_remove) > 0:
                        for element in elements_to_remove:
                            # check is element is in
                            if element in row_clean:
                                row_clean.remove(element)
                            else:
                                pass
                        try:
                            indices_for_match.append(row_clean[0])
                            elements_to_remove.append(row_clean[0])
                        except:
                            # append None if nothing within caliper range that also isn't a duplicate
                            indices_for_match.append(None)

                    else:
                        try:
                            indices_for_match.append(row_clean[0])
                            elements_to_remove.append(row_clean[0])
                        except:
                            indices_for_match.append(None)

                # if replacement is True:
                else:
                    # try and append from list (if caliper size ok)
                    try:
                        indices_for_match.append(row_clean[0])
                    # otherwise append None
                    except:
                        indices_for_match.append(None)

        # if no caliper:
        else:
            indices_for_match = []
            elements_to_remove = []
            # loop through both distance and indexes from knn simultaneously:
            for dist, row in zip(distances[:, :], indexes[:, :]):
                # convert dist and row_ids to lists:
                dist = np.ndarray.tolist(dist)
                row = np.ndarray.tolist(row)
                row_clean = row
                # check to see if replacement is False:
                if replacement == False:
                    # if there are elements in elements to remove:
                    if len(elements_to_remove) > 0:
                        for element in elements_to_remove:
                            # check is element is in row clean
                            if element in row_clean:
                                row_clean.remove(element)
                            else:
                                pass
                        try:
                            indices_for_match.append(row_clean[0])
                            elements_to_remove.append(row_clean[0])
                        except:
                            # append None if nothing within caliper range that also isn't a duplicate
                            indices_for_match.append(None)
                    # if there are NO elements in elements to remove:
                    else:
                        try:
                            indices_for_match.append(row_clean[0])
                            elements_to_remove.append(row_clean[0])
                        except:
                            indices_for_match.append(None)
                # if replacement is True:
                else:
                    # try and append from list (if caliper size ok)
                    try:
                        indices_for_match.append(row_clean[0])
                    # otherwise append None
                    except:
                        indices_for_match.append(None)

        ID_match = []
        for idxxx in indices_for_match:
            # try to pull original patient indexes specified at outset
            try:
                match = major_pred_rstindx.loc[idxxx, self.indx]
                ID_match.append(match)
            # otherwise append none to list
            except:
                ID_match.append(np.nan)

        indexes_nonull = list(filter(None, indices_for_match))
        major_matched = major_pred_rstindx.take(indexes_nonull)

        if drop_unmatched == True:
            if len(indexes[:, 1]) == len(list(filter(None, indices_for_match))):
                pass
            else:
                warnings.warn('Some values do not have a match. These are dropped for purposes of establishing a matched dataframe, and subsequent calculations and plots (effect size). If you do not wish this to be the case please set drop_unmatched=False')

            minor_pred_rstindx['matched_ID'] = ID_match
            matched_ids = minor_pred_rstindx[[self.indx, 'matched_ID']]
            matched_ids_nona = matched_ids.dropna()
            indices_to_take_from_minor_class = list(
                matched_ids_nona.index.values)
            minor_pred_rstindx = minor_pred_rstindx.take(
                indices_to_take_from_minor_class)
            self.df_matched = pd.concat(
                [minor_pred_rstindx, major_matched], axis=0, ignore_index=True)
            self.matched_ids = matched_ids_nona.reset_index(drop=True)

        else:
            if len(indexes[:, 1]) == len(list(filter(None, indices_for_match))):
                pass
            else:
                warnings.warn(
                    'Some values do not have a match. These are not dropped for purposes of subsequent calculation and plots. If you do not wish this to be the case please set drop_unmatched=True')

            self.df_matched = pd.concat(
                [minor_pred_rstindx, major_matched], axis=0, ignore_index=True)
            minor_pred_rstindx['matched_ID'] = ID_match
            self.matched_ids = minor_pred_rstindx[[self.indx, 'matched_ID']]

    def knn_matched_12n(self, matcher='propensity_logit', how_many=1):
        """
        knn_matched_12n -- Match data using k-nn algorithm to sample 1:selected_number
        Parameters
        ----------
        matcher : str
           string that will used to match - propensity score or proppensity logit
        how_many : int
            integer to indicate how many times you want to pull matches for the minor class
        Returns
        matched_ids : pd.DataFrame
            DataFrame with column with matched ID based on k-NN algorithm
        """
        # set matcher to matcher
        matcher = matcher
        # allocate treatment and control dfs:
        predicted_data_control = self.predicted_data[self.predicted_data[self.treatment] == 0]
        predicted_data_treatment = self.predicted_data[self.predicted_data[self.treatment] == 1]
        # determine which is the major class:
        if len(predicted_data_treatment) < len(predicted_data_control):
            min_pred, major_pred = predicted_data_treatment, predicted_data_control
            major_pred_rstindx = major_pred.reset_index(drop=True)
            minor_pred_rstindx = min_pred.reset_index(drop=True)
        elif len(predicted_data_treatment) > len(predicted_data_control):
            min_pred, major_pred = predicted_data_control, predicted_data_treatment
            major_pred_rstindx = major_pred.reset_index(drop=True)
            minor_pred_rstindx = min_pred.reset_index(drop=True)
        else:
            min_pred, major_pred = predicted_data_control, predicted_data_treatment
            major_pred_rstindx = major_pred.reset_index(drop=True)
            minor_pred_rstindx = min_pred.reset_index(drop=True)
        major_pred_rstindx_og = major_pred_rstindx.copy()

        #self.indices_for_match = []
        ID_match = []
        # indexes for match needs to keep all those tagged from all loops so must be instantiated outside the loop
        for many in range(how_many):
            # need to fit KNN with larger class
            knn = NearestNeighbors(n_neighbors=len(major_pred_rstindx), p=2)
            knn.fit(major_pred_rstindx[[matcher]].to_numpy())
            distances, indexes = knn.kneighbors(
                minor_pred_rstindx[[matcher]].to_numpy(), n_neighbors=len(major_pred_rstindx))
#             self.distances = distances
#             self.indexes = indexes

            # lists to remove elements that have already matched
            elements_to_remove = []
            indices_in_loop = []
            # loop through both distance and indexes from knn simultaneously:
            for dist, row in zip(distances[:, :], indexes[:, :]):
                # convert dist and row_ids to lists:
                dist = np.ndarray.tolist(dist)
                row = np.ndarray.tolist(row)
#                 self.dist_indices = [idx for idx, element in enumerate(
#                     dist)]
                # clean up the ids from the distance exclusion:
                #self.row_clean = row
                if len(elements_to_remove) > 0:
                    for element in elements_to_remove:
                        # check is element is in
                        if element in row:
                            row.remove(element)
                        else:
                            pass
                    elements_to_remove.append(row[0])
                    indices_in_loop.append(row[0])
                # if there are NO elements in elements to remove:
                else:
                    elements_to_remove.append(row[0])
                    indices_in_loop.append(row[0])

            # remove all used indexes from this round of KNN and prepare for next with the last set removed:
            # create df for exclusion on those that have matched
            major_matched_loop = major_pred_rstindx.take(indices_in_loop)
            # pull patient indexes (self.indx) from dataframe as we go (since array values will repeat):
            for idxxx in indices_in_loop:
                match = major_pred_rstindx.loc[idxxx, self.indx]
                ID_match.append(match)

#             # run the exclusion for the next run:
            major_pred_rstindx = pd.merge(major_pred_rstindx, major_matched_loop[[
                                          self.indx]], on=self.indx, how="outer", indicator=True).query('_merge=="left_only"')
            major_pred_rstindx = major_pred_rstindx.drop(['_merge'], axis=1)
            # reset index for next round of matching
            major_pred_rstindx = major_pred_rstindx.reset_index(drop=True)

        # build dataframe passed back to user
        df_2_extract_major_class = pd.DataFrame(ID_match, columns=[self.indx])
        df_4_user = pd.merge(
            major_pred_rstindx_og, df_2_extract_major_class, on=self.indx, how="right")

        self.df_matched = pd.concat(
            [minor_pred_rstindx, df_4_user], axis=0, ignore_index=True)

        # build df of matched ids passed back to user:
        # create array for easy viewing of the 'n' levels of macthing performed
        ID_match_array = np.asarray(ID_match).reshape(
            (len(minor_pred_rstindx), how_many))
        major_list_cols = []
        for many in range(how_many):
            col_name = 'largerclass_' + str(many) + 'group'
            major_list_cols.append(col_name)
        # convert matched ids from major class to df
        major_class_matched_ids_df = pd.DataFrame(
            ID_match_array, columns=major_list_cols)
        self.matched_ids = pd.concat(
            [minor_pred_rstindx[[self.indx]], major_class_matched_ids_df], axis=1)

    def plot_match(self, matched_entity='propensity_logit', Title='Side by side matched controls', Ylabel='Number of patients', Xlabel='propensity logit', names=['treatment', 'control'], colors=['#E69F00', '#56B4E9'], save=False):
        """
        knn_matched -- Match data using k-nn algorithm
        Parameters
        ----------
        matched_entity : str
           string that will used to match - propensity_score or proppensity_logit
        Title : str
           Title of plot
        Ylabel : str
           Label for y axis
        Xlabel : str
           Label for x axis
        names  : list
           List of 2 groups
        colors : str
           string of hex code for group 1 and group 2
        save   : Bool
            Whether to save the figure in pwd (default = False)
        Returns
        grpahic
        """
        dftreat = self.df_matched[self.df_matched[self.treatment] == 1]
        dfcontrol = self.df_matched[self.df_matched[self.treatment] == 0]
        x1 = dftreat[matched_entity]
        x2 = dfcontrol[matched_entity]
        # Assign colors for each airline and the names
        colors = colors
        names = names
        sns.set_style("white")
        # Make the histogram using a list of lists
        # Normalize the flights and assign colors and names
        plt.hist([x1, x2], color=colors, label=names)
        # Plot formatting
        plt.legend()
        plt.xlabel(Xlabel)
        plt.ylabel(Ylabel)
        plt.title(Title)
        if save == True:
            plt.savefig('propensity_match.png', dpi=250)
        else:
            pass

    def effect_size_plot(self, title='Standardized Mean differences accross covariates before and after matching',
                         before_color='#FCB754', after_color='#3EC8FB', save=False):
        """
        effect_size_plot -- Plot effect size on each variable before and after matching
        Parameters
        ----------
        title : str (optional)
           Title the graphic generated
        before_color : str (hex)
           color for the before matching effect size per variable
        after_color : str (hex)
           color for the after matching effect size per variable
        save : bool
            Save graphic or not (default = False)
        Returns
        seaborn graphic
        """
        df_preds_after = self.df_matched[[self.treatment] + self.xvars]
        df_preds_b4 = self.data[[self.treatment] + self.xvars]
        df_preds_after_float = df_preds_after.astype(float)
        df_preds_b4_float = df_preds_b4.astype(float)

        data = []
        for cl in self.xvars:
            data.append([cl, 'before', cohenD(
                df_preds_b4_float, self.treatment, cl)])
            data.append([cl, 'after', cohenD(
                df_preds_after_float, self.treatment, cl)])
        self.effect_size = pd.DataFrame(
            data, columns=['Variable', 'matching', 'Effect Size'])
        sns.set_style("white")
        sns_plot = sns.barplot(data=self.effect_size, y='Variable', x='Effect Size', hue='matching', palette=[
            before_color, after_color], orient='h')
        sns_plot.set(title=title)
        if save == True:
            sns_plot.figure.savefig(
                'effect_size.png', dpi=250, bbox_inches="tight")
        else:
            pass
