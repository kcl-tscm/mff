# Tutorials

comming soon.

Latex support:
- inline: $\theta$,

\begin{equation}
   (a + b)^2 = a^2 + 2ab + b^2
   (a - b)^2 = a^2 - 2ab + b^2
\end{equation}

List:
- a
- b

1. a
2. b


## Subsection

recommonmark parser for Docutils and Sphinx
```math
display
```
and ``$ y=\sum_{i=1}^n g(x_i) $``

MathJax

Configurable delimiters — by default \(y=\sum_{i=1}^n g(x_i)\), $$display$$, \[display\]. Smart enough to parse $y = x^2 \hbox{ when $x > 2$}$. as one expression. Supports backslash-escaping to prevent parsing as math.

Recognizes \begin{foo}...\end{foo} without any extra delimiters. Supports macros — recognizes \def..., \newcommand... without any extra delimiters.

Can also support MathML and AsciiMath — depends on configuration.

KaTeX
Much faster (and smaller) than MathJax but supports considerably less constructs so far. Doesn't yet recognize math by any delimiter, need to call function per math fragment. Notable for outputting stable HTML+CSS+webfont that work on all modern browsers, so can run during conversion and work without javascript on client. (MathJax 2.5 also has "CommonHTML" output but it's bad quality, only usable as preview; the closest server-side option is MathJax-node outputing SVG.)

Not yet used much with markdown.

ikiwiki with mathjax plugin
Both display ($$y=\sum_{i=1}^n g(x_i)$$ or [y=\sum_{i=1}^n g(x_i)]) and inline ($y=\sum_{i=1}^n g(x_i)$ or (y=\sum_{i=1}^n g(x_i))) math are supported. Just take care that inline math must stay on *one line in your markdown source. A single literal dollar sign in a line does not need to be escaped, but two do.

