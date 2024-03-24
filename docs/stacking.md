# How to improve performance
stacking involves taking multiple prediction models and using them as input for a final, combiner model. This final model aims to learn the best way to blend the individual models' predictions to improve overall accuracy.


## Creates a diverge set of strong models
One should work on creating several strong models. Each should be quite good at the task one am trying to achieve.

## Deciding what models to combine
After creating many strong models one should check the how different predictions the models gives as if they are very good but there is a large difference in predictions that means they have learned patterns that the other do not know. Then by comining them one can improve the total predictive power of the model. One can se how different two models are by observing there residual values.

## Combining of models

There are several ways of combining the different estimators. The simplest version that often works well is running all the different models and taking a weighted average of the different models, where the better models should have more of a say of the final output (larger weight) then less performant models. 