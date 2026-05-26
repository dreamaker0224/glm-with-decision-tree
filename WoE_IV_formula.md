# WoE formula
$$WOE_{ij} = \ln \frac{Among\ Yes\ poppulation,\ \%\ of\ x_i\ with\ jth\ level}{Among\ No\ poppulation,\ \%\ of\ x_i\ with\ jth\ level} = \ln \frac{p(x_{ij}|Y = Yes)}{p(x_{ij}|Y = No)}$$

# IV formula
$$IV_i = \sum_{all\ j} (p(x_{ij}|Y = Yes) - p(x_{ij}|Y = No)) \times WOE_{ij}$$