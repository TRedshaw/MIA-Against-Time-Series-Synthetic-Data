#
#
#
# Old version of trying to get scipy.state.gaussian_kde to work when it actually went into a singular cov matrix.
#
#
#

import numpy as np
from scipy import stats


class GaussianKde(stats.gaussian_kde):
    """
    Drop-in replacement for gaussian_kde that adds the class attribute EPSILON
    to the covmat eigenvalues, to prevent exceptions due to numerical error.
    """

    # TODO Change this accordingly
    EPSILON = 1e-10  # adjust this at will

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            self._data_covariance = np.atleast_2d(np.cov(self.dataset, rowvar=1,
                                                         bias=False,
                                                         aweights=self.weights))
            # we're going the easy way here
            self._data_covariance += self.EPSILON * np.eye(len(self._data_covariance))
            self._data_inv_cov = np.linalg.inv(self._data_covariance)
            self._data_cho_cov = np.linalg.cholesky(self._data_covariance)
            #                                           ,lower=True) # ADDED CHOCOV TO ONLINE VERSION

        self.covariance = self._data_covariance * self.factor ** 2
        # self.inv_cov = self._data_inv_cov / self.factor ** 2
        L = np.linalg.cholesky(self.covariance * 2 * np.pi)
        self._norm_factor = 2 * np.log(np.diag(L)).sum()  # needed for scipy 1.5.2
        self.log_det = 2 * np.log(np.diag(L)).sum()  # changed var name on 1.6.2

        self.cho_cov = (self._data_cho_cov * self.factor).astype(np.float64) # ADDED CHOCOV TO ONLINE VERSION
