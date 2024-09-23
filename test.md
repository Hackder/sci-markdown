```python exec
keys = [1,2,3,4,5,6]
values = [0.1,0.2,0.3,0.4,0.5,0.6]
```

# This is a markdown title

This is a paragraph.
This is a second paragraph.

```python exec
table(keys, [values, values])
```

This is a paragraph.
This is a paragraph.

**Average**:
```python exec
print(round(sum(values)/len(values), 2))
```

```python exec
table(keys, [values, values])
```

$\frac{1}{2}$

```python exec
# plot values as bar graph
fig, ax = plt.subplots()
ax.bar(keys, values)
img_plot(fig)
```
