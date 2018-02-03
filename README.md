# roc_curve_test
A simple roc_curve implementation tested on different types of classifiers (discrete, probabilistic and constant)

```python
 @register("my_tokenizer")
 class MyTokenizer(Inferable, Trainable):

    def infer(*args, **kwargs):
        return self._tokenize()
    
    def _tokenize(*args, **kwargs):
        """
        Implement tokenizing here.
        """
        return tokens
 ```
