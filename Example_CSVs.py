#!/usr/bin/env python
# coding: utf-8

# ## Examples using CSV files

# Parsing comma-separtated-values (CSV) files is a common task. There are many tools available in Python to deal with this. Let's start by using the built-in `csv` module.

# [147]


import csv # using Python module


# Now, we want to open an example file, and read the contents:

# [148]


f = open('example_1.csv','rb') # use binary mode if on MS windows
d = [i for i in csv.reader(f) ] # use list comprehension to read from file
f.close() # close file
print d


# ### Adding new fields as columns

# This reads the rows of the CSV file into a list-of-lists.
# 
# Now, let's suppose we want to create columns for `last name`  and `first name` instead of having just one `name` field. The first element in the list `d` is the header, so we had the additional fields there:

# [149]


d[0].append('Last Name')
d[0].append('First Name')
print d[0]


# Now, we want to split the original `Name` field into first and last names and put these at the ends of their respective rows.

# [150]


for row in d[1:] # start at 1st, not 0th column. Each row is a list
    first,last= row[0].split() # split on white-space
    row.append(last) # append to each row
    row.append(first)
    
print d


# [151]


#%qtconsole


# ### Writing updated CSV file

# Now, we want to write out our new data in CSV format

# [152]


f = open('example_1_out.csv','wb') # write mode binary
fw = csv.writer(f) # create csv writer
fw.writerows(d) 
f.close() # close file 


# Now, opening the file `example_1_out.csv` using excel (or another reader) should show the new column. That covers the most direct and pure-Python way to dealing with CSV files. However, there are many other tools available. For example, `numpy` provides power methods to access these files.

# ## Using Numpy to parse CSV files

# Let's see how to accomplish the same work as above by using Numpy.

# [153]


import numpy as np as np 

d = np.loadtxt('example_1.csv',delimiter=',',dtype=str)

print d
print d.dtype


# Notice that we did not have to use `open` to get at the contents of the file. By default, the `delimiter` is any whitespace, so we had to change this to the comma character. The `dtype` specifies we want everything to be read in as a string. Numpy can figure out how long that string needs to be as it goes through the file so we don't have to specify that ahead of time (we probably don't know it anyway). In this case, the `dtype` turns out to be a twelve character string `S12`. Note that this is the maximum length is used for *all* strings so there is obviously a lot of extra space if most of the strings are short and just a few are long.
# 
# Numpy provides many more ways of reading data via the `dtype`. For example,

# [154]


dt = [('name',   'S64'),   # The first element of the tuple is our name for each respective column
      ('dob',    'S64'),   # and the second element is the numpy dtype we want for that column.
      ('years',  'int'),   # Here we want years as an integer, not a string
      ('degree', 'S64'), ]

d = np.loadtxt('example_1.csv',delimiter=',',dtype=dt,skiprows=1) # skip the header row
print d


# The advantage of doing it this way is that now we can compute the `years` column using `numpy` tools. For example, here is the np.mean of the years.

# [155]


print d['years'].np.mean() # using numpy np.arrays


# Now to get back to the main task at hand: splitting the name field into first and last name.

# [156]


import string

n=map(string.split,d['name'])
w=np.array([tuple(i)+tuple(j) for i,j in zip(d,n)],    # list comprehension glues tuple-ized rows together
         dtype=dt+[('first','S64'),('last','S64')]) # append new dtypes to existing list of dtypes


# That was kind of non-simple, but now we can write this to a CSV using `savetxt`.
# 

# [157]


# the comments are set to '' to avoid hash marks on the first line.
np.savetxt('np_output.csv',w,delimiter=',',fmt='%s',header='name,dob,years,degree,first,last',comments='')


# Now, you can inspect the so-generated file and verify it is a CSV.

# ## Using pandas to parse CSV files

# `pandas` is the real power tool for this job.

# [158]


import pandas as pd

d = pd.read_csv('example_1.csv')
print d
print type(d)


# Now, we have read the CSV file as a `pandas` DataFrame which is a super-structure that sits on top of `numpy`. Let's examine the columns of this DataFrame.
# 

# [159]


print d.columns


# Notice that there is an extra space after the `Name `. This potentially makes it hard to access the columns using pandas slicing. For example,

# [161]


print d.DOB # this works great when the column header name has no spaces in it.
print d['Name'] # you can also refer to columns using this syntax


# Luckily, this is not hard to fix. We just need to create another column that is free of these trailing spaces:

# [ ]


d['name']=d['Name '] # easily create extra column
print d.name         # now you can access this column using this syntax


# Pandas is a lot more powerful than this! We can parse the columns by types individually by providing a `dtype` for each column as a dictionary.

# [ ]


d = pd.read_csv('example_1.csv',dtype={'Name ':'S64','DOB':'S64','Years':int,'Degree':'S64'})
print d


# Now, we can compute along the columns as we did before with `numpy`.

# [ ]


print d.Years.np.mean()


# You can also parse the `DOB` field to get a true timestamp instead of a string using the `parse_dates` keyword.

# [171]


d = pd.read_csv('example_1.csv',dtype={'Name':'S64',
                                       'DOB':'S64',
                                       'Years':int,
                                       'Degree':'S64'},parse_dates=[1])


# Now, we can compute with these `datetime` objects as in the following.

# [164]


# difference in birthdays between Alice Jones and John Book
print d.DOB[0] -  d.DOB[2]


# Now we now how many days are between the respective birthdays of Alice Jones and John Book.

# [ ]


get_ipython().run_line_magic('qtconsole', '')


# [166]


d['first']=map(lambda x:string.split(x)[0],d['Name'])
d['last']=map(lambda x:string.split(x)[1],d['Name'])
print d


# [167]


print d


# ## ject into a sqlite database

# [170]


import pandas.io.sql as pd_sql
import sqlite3 as sql # sqlite3 is built into Python

con = sql.connect("example_1.db")
pd_sql.write_frame(d,'data',con) # write to DB as table named "data"
con.close()


# [ ]




