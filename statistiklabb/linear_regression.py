import numpy as np
from scipy.stats import f, t, pearsonr

class LinearRegression:
    def __init__(self, X, Y, confidence_level=0.95):
        self.X = np.column_stack((np.ones(len(X)), X))  # Lägg till intercept-term
        self.Y = Y
        self.n, self.d = self.X.shape   # Stickprovsstorlek och antal variabler (inklusive intercept)
        self.confidence_level = confidence_level
        self.b = self._calculate_coefficients()  # Regressionskoefficienter
        self.sigma2 = self._calculate_variance()  # Varians
        self.cov_matrix = self._calculate_covariance_matrix()  # Kovariansmatris

    def _calculate_coefficients(self):
        """Beräkna regressionskoefficienter med OLS."""
        return np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y

    def _calculate_variance(self):
        """Beräkna den obiaserade variansskattningen."""
        residuals = self.Y - self.X @ self.b
        sse = np.sum(residuals**2)
        return sse / (self.n - self.d)

    def _calculate_covariance_matrix(self):
        """Beräkna varians-/kovariansmatrisen för koefficienterna."""
        return np.linalg.inv(self.X.T @ self.X) * self.sigma2

    def standard_deviation(self):
        """Beräkna standardavvikelsen."""
        return np.sqrt(self.sigma2)

    def significance_of_regression(self):
        """Testa regressionens signifikans med F-fördelningen."""
        sse = np.sum((self.Y - self.X @ self.b)**2)
        syy = np.sum((self.Y - np.mean(self.Y))**2)
        ssr = syy - sse
        f_statistic = (ssr / (self.d - 1)) / self.sigma2
        p_value = f.sf(f_statistic, self.d - 1, self.n - self.d)
        return f_statistic, p_value

    def r_squared(self):
        """Beräkna determinationskoefficienten (R^2)."""
        sse = np.sum((self.Y - self.X @ self.b)**2)
        syy = np.sum((self.Y - np.mean(self.Y))**2)
        return 1 - (sse / syy)

    def significance_of_parameter(self, i):
        """TTesta signifikansen för en enskild parameter med T-fördelningen."""
        t_statistic = self.b[i] / (np.sqrt(self.sigma2 * self.cov_matrix[i, i]))
        p_value = 2 * min(t.cdf(t_statistic, self.n - self.d), 1 - t.cdf(t_statistic, self.n - self.d))
        return t_statistic, p_value

    def confidence_interval(self, i):
        """Beräkna konfidensintervallet för en enskild parameter."""
        t_alpha = t.ppf(1 - (1 - self.confidence_level) / 2, self.n - self.d)
        margin_of_error = t_alpha * np.sqrt(self.sigma2 * self.cov_matrix[i, i])
        return self.b[i] - margin_of_error, self.b[i] + margin_of_error

    def pearson_correlation(self):
        """Beräkna Pearsons korrelation mellan alla parameterpar."""
        correlations = np.zeros((self.d, self.d))
        for i in range(self.d):
            for j in range(self.d):
                correlations[i, j], _ = pearsonr(self.X[:, i], self.X[:, j])
        return correlations