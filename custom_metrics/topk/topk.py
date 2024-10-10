"""Top K metric."""

import datasets
import numpy as np
from sklearn.metrics import top_k_accuracy_score
import evaluate

_DESCRIPTION = """
Top K is the proportion of correct predictions (best 5) among the total number of cases processed.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions (`list` of `int`): Predicted labels.
    references (`list` of `int`): Ground truth labels.
    normalize (`boolean`): If set to False, returns the number of correctly classified samples. 
    Otherwise, returns the fraction of correctly classified samples. Defaults to True.
    sample_weight (`list` of `float`): Sample weights Defaults to None.

Returns:
    topK (`float` or `int`): Accuracy score. Minimum possible value is 0. Maximum possible value is 1.0, 
    or the number of examples input, if `normalize` is set to `True`. A higher score means higher accuracy.
"""

_CITATION = """
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
"""


class Topk(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("float")),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.top_k_accuracy_score.html"],
        )

    def _compute(self, predictions, references, normalize=True, sample_weight=None, k=5, labels=None):

        predictions = np.array(predictions)
        references = np.array(references)

        # print(predictions.shape)
        # print(references.shape)

        return {
            f"top_{k}_accuracy": float(
                top_k_accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight, k=k,
                                     labels=labels)
            )
        }