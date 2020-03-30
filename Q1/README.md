# (Q1) The Missing Country

There was a bug in the code and for one specific country, the records don't have the country field logged. It just shows up as an empty field (""). The search sessions with a missing country either come from a country that is completely missing from the data, or from one of the countries that are logged in the data. Can you determine which country it is the most likely to be? Explain your hypothesis and the data analysis tasks you did to find the missing country.


## Solution

A preliminary inspection of the data shows that for each search session, we have the following information available:

1. searched cities
2. UNIX timestamp for that session
3. unique ID of the user
4. joining date of the user
5. the country of the user

The goal is to analyze the usefulness of the first 4 features in order to find the patterns that are useful in detecting the missing country.


### Step 1: Inspecting the Searched Cities

For each country, we plot the count of each searched city, in order to see if:

1. users from different countries have a specific preference in searching cities, and
2. if the users from the missing country follow a similar pattern to another country

![cities_per_country](Figures/cities_per_country.png)

Looking at the figures above, we see that all countries follow a similar pattern in terms of searched cities, and the missing country's users follow this pattern, too. As a result, the searched cities do not prove as a discriminative feature between different countries.


### Step 2: Inspection of the Access Times

Using the UNIX timestamp, we first extract the time within the day for each search session. Then we calculate the mean and standard deviation of the access time for the users of each country, as well as the missing country, and plot the normal distribution of them, to compare the distribution for the missing country with other countries.

![access_times](Figures/access_times.png)

The figure above shows that:

1. the distribution for the European countries is very similar and concentrated over a particular time-span, while the mean of the distribution for US is far on the right, showing that the distributions are pretty much dependent upon the time-zome of the countries;
2. the distribution for US is less concentrated than the European countries, as expected based on the width of US, which covers a wider range of time-zones.

For the missing country, it can be observed that the mean access time is 5 hours and 22 minutes ahead of UK. Based on this, we can guess that the missing country's time-zone is 5-6 hours ahead of UK's time. Also, based on the concentration of the distribution for the missing country, it can be hypothesized that the missing country should not be as wide as US, and it probably covers a single time-zone.


### Step 3) Estimation of Population

First, we check the number of users from each country, and we compare that with the population of the country (the population of the countries was collected from https://www.worldometers.info/world-population/population-by-country/). Having the number of users and the population for each given country, we would like to see if there is a correlation between the two, which can potentially help us in having an estimate of the population of the missing country.

|                | US     | DE    | UK    | UNK | FR    | ES    | IT    |
|:---------------|:------:|:-----:|:-----:|:---:|:-----:|:-----:|:-----:|
| __# of Users__ |  1129  |  1051 |  1043 | 792 |  665  |  569  |  528  |
| __Population__ | 327.2M | 82.8M | 66.4M | ??? | 67.0M | 46.7M | 60.5M |

The table above shows that, in general, the more populated a country is, the higher its number of users is. The table (having the countries sorted by the number of users) also shows that, the number of users for the unknown country is between that of UK and France. Knowing that the searched cities all belong to US and Canada, and the missing country is presumably located farther away from US and Canada compared to France (based on the access time distribution, discussed above), we can assume that the missing country is far more populated that France, to allow for such a number of users.

Checking populated countries which fall in the estimated time-zone (around 5 and a half hours ahead of UK's), makes __India__ a good candidate. But, let's first check the remaining features to see if they also support our assumption.


### Step 4) Inspection of the Days in the Week for Sessions

Below, the proportion of search sessions per week day for each country is plotted. We would like to see if users from different countries have a preferred weekday to do their search.

![access_days](Figures/access_days.png)

Looking at the figures above, it can be seen that, there is not a significant difference in the proportion of searches done in different weekdays. As a result, this cannot be used as a discriminative feature.


### Step 5) Inspection of Joining Days

Assuming that joining the platform requires more time (as the user is not yet familiar with the platform, and the registration could take some time), the users might show a more significant preference towards specific weekdays to join the platform than to just access it (inspected in step 4)

![joining_days](Figures/joining_days.png)

The figures above show that, in fact, when joining the platform, users show a preference towards some days of the week. Looking at the figures for the 6 given countries, we can see that the users generally prefer to join the platform on the weekends (Saturday and Sunday) and the first two working days of the week.

The missing country follows a similar pattern, based on which we can guess that the weekends in the missing country is also Saturday and Sunday. This is in line with our hypothesis that __India__ is the missing country: a populated country which is around 5 and a half hours ahead of the UK, and has Saturdays and Sundays as the weekends.