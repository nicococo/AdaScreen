### NEWS
- This page, as well as the corresponding code on GitHub,
is currently undergoing major changes and likely not to work properly.
Please come back later... :)
- Wiki can be found [here](https://github.com/nicococo/AdaScreen/wiki)

### About the AdaScreen Software Package
We developed our screening framework in the _Python_ programming language
and it can be freely downloaded using the link above or conveniently
downloaded and installed automatically, using the _pip_ command.

It is designed to efficiently implement various screening rules without
changing the lasso path solver (e.g., scikit-learn lasso solver).
Even though different screening rules require different constraints
and equations, they all share common data structures; thus, we wrap all
of them into a single framework. An advantage of this approach is that
the lasso path solver needs to interact with only one abstract class for
screening rules.

### AdaScreen Structure
To systematically manage data structures involved in screening, we divide
them into _Globals_ and _Locals_, where _Globals_ refer to variables that
do not change over the lambda path (e.g., the inputs **X**, **y**, lambda_max).
In contrast, _Locals_ refer to variables that change over the lambda path
(e.g., the last regularization parameter lambda_0 or the solution beta^*(lambda_0)
from the previous lambda_0). We maintain these data structures in the path solver,
and call the _init_ method in the _ScreeningRule_ before entering the
main iteration loop of our screening framework.

Furthermore, we designed our screening framework in such a way that
all screening rules can be derived from the abstract base class. For example,
many screening rules can be framed with a single sphere constraint,
consisting of a center and a radius. In such cases, screening rules can be
implemented by overloading the _getSphere_ function. For more advanced methods,
corresponding functions need to be overloaded. Now, let us consider how AdaScreen
can be instantiated under our framework. _AdaScreen_ maintains a list of
_ScreeningRules_ itself and can return global and local half-space constraints.
Therefore, it is easy to implement AdaScreen with any sphere and any half-space
constraints. For example, to implement AdaScreen with EDPP sphere constraint
and Sasvi local half-space constraint, we first instantiate _EDPP_, _Sasvi_,
and _AdaScreen_. Then in _AdaScreen_, we simply call _setSphereRule(EDPP)_
and _addHalfspace(Sasvi)_.

### About the AdaScreen Screening Ensemble
In order to solve large-scale lasso problems, screening algorithms have been
developed that discard features with zero coefficients based on a computationally
efficient screening rule. Most existing screening rules were developed from
a spherical constraint and half-space constraints on the dual optimum. However,
existing rules admit at most two half-space constraints due to the computational
cost incurred by the half-spaces, even though additional constraints may be useful
to discard more features. In this paper, we present AdaScreen, an adaptive lasso
screening rule ensemble, which allows to combine any one sphere with multiple
half-space constraints on a dual optimal solution. Thanks to geometrical considerations
that lead to a simple closed form solution for AdaScreen, we can incorporate a large
number of half-space constraints at small computational cost. In our experiments,
we show that AdaScreen with multiple half-space constraints simultaneously improves
screening performance and speeds up lasso solvers.

### Disclaimer
When using the software, or parts of it, please cite:
_to appear_

### Authors
Method: Seunghak Lee, Nico Goernitz, Eric P. Xing, David Heckerman, Christoph Lippert
Software: Nico Goernitz & Seunghak Lee