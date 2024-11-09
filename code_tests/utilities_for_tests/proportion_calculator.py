from math import sqrt

from scipy.stats import norm


class ProportionStatCalculator:
    def __init__(self, number_of_successes: int, number_of_trials: int):
        self.number_of_successes: int = number_of_successes
        self.number_of_trials: int = number_of_trials

    def determine_if_population_proportion_is_above_p0(
        self, p0: float, desired_confidence: float
    ) -> tuple[float, bool, str]:
        """
        Requirements
        - Simple random sample
        - Binomial distribution conditions are satisfied
        - np >= 5 and nq >= 5 where n is the number of trials, p is the population proportion, and q is 1 - p

        Input
        - p0: float - the population proportion that is being tested
        - desired_confidence: float - the desired confidence level

        Output
        - p_value: float - the p-value of the test
        - hypothesis_rejected: bool - whether the null hypothesis is rejected
        - written_conclusion: str - a written conclusion of the test

        Useful values
        - A sample success of 98/100 (98%) needed to show that proportion is above 95% with 90% confidence
        - A sample success of 194/200 (97%) needed to show that proportion is above 95% with 90% confidence
        - A sample success of 27/30 (90%) needed to show that proportion is above 80% with 90% confidence
        - A sample success of 86/100 (86%) needed to show that proportion is above 80% with 90% confidence

        For an online test Calculator see the below (it may or may not use t distribution at lower sample sizes)
        https://www.statssolver.com/hypothesis-testing.html
        """

        if (
            self.number_of_trials * p0 < 5
            or self.number_of_trials * (1 - p0) < 5
        ):
            raise ValueError(
                "The normal distribution approximation conditions are not satisfied. Too few samples given the desired p0 to test"
            )

        sample_proportion = self.number_of_successes / self.number_of_trials

        if (
            self.number_of_trials * sample_proportion < 5
            or self.number_of_trials * (1 - sample_proportion) < 5
        ):
            raise ValueError(
                "The normal distribution approximation conditions are not satisfied. Sample proportion is too close to 0 or 1"
            )

        standard_error = sqrt(p0 * (1 - p0) / self.number_of_trials)
        z_score = (sample_proportion - p0) / standard_error

        # Since we're testing 'larger', find the area to the right of the z-score
        p_value: float = float(1 - norm.cdf(z_score))

        alpha = 1 - desired_confidence
        hypothesis_rejected = p_value < alpha

        if hypothesis_rejected:
            written_conclusion = f"If the null hypothesis is true (the proportion is equal to and not greater than {p0*100:.2f}%), then there is a {p_value*100:.2f}% probability that the sample (estimated) proportion is {sample_proportion*100:.2f}% or more (the success percentage found in the sample). Thus, we reject the null hypothesis with {desired_confidence*100:.2f}% confidence since at the {alpha*100:.2f}% level of significance, the sample data do give enough evidence to conclude that the proportion is greater than {p0*100:.2f}%. There were {self.number_of_successes} successes in {self.number_of_trials} trials."
        else:
            written_conclusion = f"If the null hypothesis is true (the proportion is equal to and not greater than {p0*100:.2f}%), then there is a {p_value*100:.2f}% probability that the sample (estimated) proportion is {sample_proportion*100:.2f}% or more (the success percentage found in the sample). Thus, we fail to reject the null hypothesis since at the {alpha*100:.2f}% level of significance, the sample data do not give enough evidence to conclude that the proportion is greater than {p0*100:.2f}%. There were {self.number_of_successes} successes in {self.number_of_trials} trials."
        return p_value, hypothesis_rejected, written_conclusion
