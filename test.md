```python exec
keys = [1,2,3,4,5,6]
values = np.random.random(6) * 10

def median(vals):
    sorted_vals = sorted(vals)
    if len(sorted_vals) % 2 == 0:
        return (sorted_vals[len(sorted_vals)//2] + sorted_vals[len(sorted_vals)//2 - 1])/2
    else:
        return sorted_vals[len(sorted_vals)//2]
```

# This is a markdown title

This is a paragraph..
This is a second paragraph.

```python exec
rtable(keys, [values, values])
```

This is a paragraph.
This is a paragraph.

**Average**:
```python exec
pprint(sum(values)/len(values))
1/0
```

**Median**:
```python exec
pprint(median(values))
```

```python exec
ctable(keys, [values, values])
```

$\frac{1}{2}$

```python exec
# plot values as bar graph
fig, ax = plt.subplots()
ax.bar(keys, values)
img_plot(fig)
```
