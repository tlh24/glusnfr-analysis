# Description

[Click here for the GluSnFr mutation analysis tool!](https://tlh24.github.io/glusnfr-analysis/sniffer.html)

This is a graphical representation for seeing how AA substitutions affect sensor properties, and was used to aid selection of mutations during the development of GluSnFR3.   Please see the main text for full details.

## Left Panel

The left panel is a 2D plot of variant's measured quantities.  You can select the X and Y axes by clicking on them; hover over the labels for a tooltip for a longer description of the quantity. 

In the 2D plot, each variant is a colored circle.  Color refers to Kd; blue = 0 um, red = 1 mM.  This mapping is linear, but you can hover over each circle to see the numerical value of the binding affinity. 

- Click on a variant circle to reveal the network of single (gray line with arrow) and double (cyan line) amino-acid substitutions that relate it to other screened variants.  This will highlight the variant with a red border. 
- Hover over a mutation line to see its identity, e.g. 'G33S'.  This will also highlight entries in the right panel (see below.)
- Click on a mutation line to show all other instances where that mutation occurs in the data set, and the resulting effect on the plotted measured quantities.  For example, G33S seems to decrease Kon, and increase delta F / F, with significant dependence on background. 
- Click on this mutation line again to hide all the single or double substitutions.
- Likewise, click on the red-highlighted variant to turn off its web of mutations. 
- If the network of mutations gets too messy, please reload the page. 


## Right Panel

Here the properties of individual AA substitutions are plotted relative to their effect on three quantities: delta F / F, Kd, Kon, and a synthetic quantity 'fitness'.  Fitness is a saturating function of Kon and delta F / F, as shown in the formula. 

To generate this plot, first a binary matrix of all substitutions is formed, where the column indexes variant, and the row indexes the presence (1) or absence (0) of a particular AA at one location (e.g. 33S). 
This binary matrix is sub-sampled (rows and columns are removed) to take 80% of the measured sequences and 60% of the measured AA substitutions.  The resulting matrix is fit via SVD to linearly predict the dependent variable (e.g. delta F / F)

This sub-sampling is done randomly for 50k times, and if SVD converges, the linear regression weights are recorded. 

The mean and standard deviation of these linear regression weights, per AA substituion, are sorted and plotted in the right panel.  (Weights that fall below a z-score threshold are removed.) Hence, if the presence of an AA at a location is generally predictive of a change in measured quantity, independent of other background changes, it will appear in the respective cascade. 

Note that this bootstrap regression is a heuristic means of *ordering* the AA substitutions, not an actual *measure of effect*. 

- Hover over circles to see how many sequences have that AA, the z-scored estimate of effect, and the mean +- std. of the weight.  For example, "34G" has a positive effect on delta F / F in 4 out of 193 screened variants.  Hovering shows the AA substitutions in the cascades for other dependent variables, too. 
- Click on the circles to highlight in red the variants that posses this AA substitution, as well as all the mutation lines that confer the substitution.
- Click again to hide the highlight and mutation lines. 
- Back on the left panel, click on a variant circle to show all the AA substitutions that it has, as well as their estimated effect on the measured quantities.

## Misc

The variant names are from our internal screens, and otherwise unrelated to anything else. (They can be ordered by time in the left panel, though, by clicking on "Var#")

Feel free to fork or clone the repository: https://github.com/tlh24/glusnfr-analysis

This analysis has also been performed for GCaMP family of sensors; if you are interested in that, please email the authors. 


