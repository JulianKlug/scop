import warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import Counter
from sklearn.model_selection import train_test_split as split

class SortedStratifiedKFold(StratifiedKFold):
    '''Stratified K-Fold cross validator.
    from: https://github.com/biocore/calour/blob/master/calour/training.py#L144

    Please see :class:`sklearn.model_selection.StratifiedKFold` for
    documentation for parameters, etc. It is very similar to that class
    except this is for regression of numeric values.
    This implementation basically assigns a unique label (int here) to
    each consecutive `n_splits` values after y is sorted. Then rely on
    StratifiedKFold to split. The idea is borrowed from this `blog
    <http://scottclowe.com/2016-03-19-stratified-regression-partitions/>`_.
    See Also
    --------
    RepeatedSortedStratifiedKFold
    '''
    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _sort_partition(self, y):
        n = len(y)
        cats = np.empty(n, dtype='u4')
        div, mod = divmod(n, self.n_splits)
        cats[:n-mod] = np.repeat(range(div), self.n_splits)
        cats[n-mod:] = div + 1
        # run argsort twice to get the rank of each y value
        return cats[np.argsort(np.argsort(y))]

    def split(self, X, y, groups=None):
        y_cat = self._sort_partition(y)
        return super().split(X, y_cat, groups)

    def _make_test_folds(self, X, y=None):
        '''The sole purpose of this function is to suppress the specific unintended warning from sklearn.'''
        with warnings.catch_warnings():
            # suppress specific warnings
            warnings.filterwarnings("ignore", message="The least populated class in y has only 1 members, which is less than n_splits=")
            return super()._make_test_folds(X, y)


def estimate_nbins(y):
    """
    Break down target vartiable into bins.
    Args:
        y (pd.Series): stratification target variable.
    Returns:
        bins (array): bins' values.
    """
    if len(y)/10 <= 100:
        nbins = int(len(y)/10)
    else:
        nbins = 100
    bins = np.linspace(min(y), max(y), nbins)
    return bins

def combine_single_valued_bins(y_binned):
    """
    Correct the assigned bins if some bins include a single value (can not be split).
    Find bins with single values and:
        - try to combine them to the nearest neighbors within these single bins
        - combine the ones that do not have neighbors among the single values with
        the rest of the bins.
    Args:
        y_binned (array): original y_binned values.
    Returns:
        y_binned (array): processed y_binned values.
    """
    # count number of records in each bin
    y_binned_count = dict(Counter(y_binned))
    # combine the single-valued-bins with nearest neighbors
    keys_with_single_value = []
    for key, value in y_binned_count.items():
        if value == 1:
            keys_with_single_value.append(key)

    # first combine with singles in keys_with_single_values
    def combine_singles(val, lst, operator, keys_with_single_value):
        for ix, v in enumerate(lst):
            if v == val:
                combine_with = lst[ix] + 1 if operator == 'subtract' else lst[ix] - 1
                y_binned[y_binned == val] = combine_with
                keys_with_single_value = [x for x in keys_with_single_value if x not in [val, combine_with]]
                y_binned_count[combine_with] = y_binned_count[combine_with] + 1 if operator == 'subtract' else y_binned_count[combine_with] - 1
                if val in y_binned_count.keys():
                    del y_binned_count[val]
        return keys_with_single_value
    for val in keys_with_single_value:
        # for each single value:
            # create lists based on keys_with_single_value with +-1 deviation
            # use these lists to find a match in keys_with_single_value
        lst_without_val = [i for i in keys_with_single_value if i != val]
        add_list = [x+1 for x in lst_without_val]
        subtract_list = [x-1 for x in lst_without_val]

        keys_with_single_value = combine_singles(val, subtract_list, 'subtract', keys_with_single_value)
        keys_with_single_value = combine_singles(val, add_list, 'add', keys_with_single_value)

    # now conbine the leftover keys_with_single_values with the rest of the bins
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    for val in keys_with_single_value:
        nearest = find_nearest([x for x in y_binned if x not in keys_with_single_value], val)
        ix_to_change = np.where(y_binned == val)[0][0]
        y_binned[ix_to_change] = nearest

    return y_binned


def continuously_stratified_train_test_split(
            *arrays,
            test_size=None,
            train_size=None,
            random_state=None,
            shuffle=True,
            stratify=None,
        ):
    """
    Create stratified splits for based on continuous data.
    for continuous target stratification binning of the target variable is performed before split.
    """
    if random_state:
        np.random.seed(random_state)

    # assign continuous target values into bins
    bins = estimate_nbins(stratify)
    y_binned = np.digitize(stratify, bins)
    # correct bins if necessary
    y_binned = combine_single_valued_bins(y_binned)

    return split(*arrays,
                 test_size=test_size,
                 train_size=train_size,
                 random_state=random_state,
                 shuffle=shuffle,
                 stratify=y_binned)


