# Full Preset Archive

`Full` has been removed from the shipped plugin UI and public docs for now.

Reason:
- it could fully register while still producing worse practical alignment than
  `Balanced`, `Standard`, or `Dense`
- several layouts were internally consistent but too redundant or low-value for
  the current capture style

Latest archived public-facing layout before removal:

```text
full | 00:12x@0/f65/start0; 01:2x@30/f65/start15; 02:2x@-30/f65/start105; ZN@90/f65
```

That layout was:
- 12 horizon
- 2 above
- 2 below
- 1 zenith

Earlier experimental layouts tested locally:

```text
full | 00:8x@0/f65/start0; 01:4x@30/f65/start22.5; 02:4x@-30/f65/start67.5;
03:4x@55/f65/start0; 04:4x@-55/f65/start45; ZN@90/f65; ND@-90/f65
```

```text
full | 00:12x@0/f65/start0; 01:4x@30/f65/start15; 02:4x@-30/f65/start60; ZN@90/f65
```

If `Full` comes back later, the most promising next direction is not "more
views everywhere", but a horizon-biased 20+ layout with shallow staggered
upper/lower belts.
