# %% [markdown]
# ## Problem Set 5

# %% [markdown]
# ### Problem 1:
# 
# a. True, we saw in class that these two methods for regressing are the exact same. 
# 
# b. True, the role of the time fixed effects variable captures effects that are specific to a certain period but across all entities in the panel regression. 
# 
# c. False, if there is no intercept present you would not have to drop one of the dummy variables that is because no multicollinearity only occurs when there is a dummy and an intercept term in the regression.
# 
# d. True, country fixed effects would be captured by using country fixed effects. 

# %% [markdown]
# ### Problem 2:

# %% [markdown]
# a. Clustered standard errors are preferable as they ensure that any residual errors correlated with the states can be accounted for in Standard error calculations, this is also accounts for heteroskedasticity robust standard errors.
# 
# b. 
# i) The effect of effect on the incumbent vote share of a one
# percentage point increase in the unemployment rate in a recession year, holding
# income constant: 0.66 percent point decrease in the incumbment vote share is the impact.
# 
# ii) What is the effect on the incumbent vote share of a one per-
# centage point increase in the unemployment rate in a non-recession year, holding
# income constant: 1.32 percent point increase in the incumbment vote share is the impact.
# 
# c. The joint f-test of UR, UR-non-recession is definitely statistically significant with a f-statistic of 4.58/0.15 = 30.533, hence unemployment rate definitely matters and is statically significant. 
# 
# d. Regression 5 captures the higher order relationship of interactions with the unemployment rate and the non-recession dummies. I would prefer regression 5 because all the variables are very statistically significant with a f-stat of 2.74/0.039 = 70.26.
# 
# e. An example of an ommitted variable that is controlled by time fixed effects could be something along the lines of the levels of interest rates that are fixed for all states on a national level at a specific time. This would be correlated with the unemployment rate (generally lower IRs mean lower inflation) and also potentially correlated with the share of votes the incumbents get (if IRs are too high you may want to change the government to lower them)
# 
# f. An example of an omitted variable that is controlled by entity fixed effects could be share of agircultural jobs. Maybe some states have a certain share of agricultural jobs and this is correlated with unemployment rate but also could be correlated with the share of votes incumbments get (maybe folks with agircultural jobs are always dissatisfied and want to vote the ruling party out).
# 
# g. The better the economic conditions the more likely the party is to vote the existing party into power. 
# 
# 

# %% [markdown]
# ## Problem 3:
# 

# %%
import pandas as pd
import statsmodels.api as sm 
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS


# %%
df = pd.read_stata('crime2.dta')
df = df.sort_values(by=['year','city'])
df

# %% [markdown]
# (a) Run a regression of crime rate (crimes per 1000 people, variable crmrte) on un-
# employment (rate in %, variable unem) on a subsample restricted to the data for
# 1982 only, then repeat the same for the 1987 subsample. Do the results make
# sense? What is the sign and magnitude of coefficients would you expect to get?

# %%
df_1982, df_1988 = df[df['year']==82], df[df['year']==87]

# %%
#1982 regression

Y = df_1982['crmrte']
X = sm.add_constant(df_1982['unem'])

model = sm.OLS(Y,X).fit(cov_type='HC1')
print(model.summary())

# %%
#1988 regression

Y = df_1988['crmrte']
X = sm.add_constant(df_1988['unem'])

model = sm.OLS(Y,X).fit(cov_type='HC1')
print(model.summary())

# %% [markdown]
# The results don't completely make sense. You would expect higher unemployment to result in higher crimerate so the beta coeffecient on unemployment should be positive. Also you would expect that the magintude of the coefficient should be relatively large and hence the coefficient should be statistically significant. In 1982 you have a positive coefficient but the coefficient is still insignificant. Further, in 1988 the coefficient on unemployment is negative and is still statistically insignificant with a relatively low value.

# %% [markdown]
# ### b.)
# An unobserved variable which varies across cities but not as much across time that could be an OMV is level of unemployment benefits. In more liberal states you may see higher unemployment benefits (this could be correlated with unemployment rate and also correlated with crime rates since if there are better unemployment benefits they people may not have to commit as many crimes).

# %% [markdown]
# ### c) Run a regression of crime rate on unemployment with city fixed effects on the full data set. Do the results change? If so, which set of results is more credible? Write explicitly what regression you are running.

# %%
# y_it = beta1 * x1_it + beta2 * x2_it + alpha_i + e_it
mod = smf.ols('crmrte ~ unem + C(city)', data=df)
res = mod.fit()
print(res.summary())

# %% [markdown]
# The results change yet unemployment is still not significant for crime rate. The explicity regression we are running is 
# 
# Crmrte_i = Beta*unem + C_I + alpha, where C_I is the dummy for each city and alpha is the intercept term.
# 

# %% [markdown]
# ### d.)
# An unobserved variable which varies across time but not over cities that could be an OMV is the interest rate. Interest rates are set on a national basis and don't change for each city. Interest rates are correlated with unemployment (by the fed's mandate), and higher interest rates could also lead to crime as it becomes harder to borrow money and therefore people may be forced to a life of crime.

# %% [markdown]
# ### (e) Do the results change when you add a time fixed effect to the regression that already has city fixed effects? If so, which set of regression results is more credible and why? Write explicitly what regression you are running

# %%
# Explicit regression: crmrte_it = intercept + beta1 * unem_it + alpha_i + theta_t + e_it
df = df.set_index(['city','year'])
mod = PanelOLS.from_formula(
    'crmrte ~ unem + EntityEffects + TimeEffects',
    data=df
)

res = mod.fit(
    cov_type='clustered',
    cluster_entity=True,
    cluster_time=True
)

print(res)

# %% [markdown]
# Now the results change signficiantly. Unemployment is now statisically signficiant. This last regression is the most reliable because it would account for OMV across time and cities, reducing OMVB and therefore leading to a better statistical inference.

# %% [markdown]
# ### f) What conclusions would you draw about the effect of unemployment on crime?

# %% [markdown]
# The higher unemployment rate the greater the impact on crime.

# %% [markdown]
# 


