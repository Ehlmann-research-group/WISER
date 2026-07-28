# Test icon fixtures

These SVGs are **purpose-built fixtures** for the `wiser.gui.theme` icon-recoloring
tests (`src/tests/test_theme.py`). They are intentionally *not* the real WISER
toolbar icons: each one isolates one of the ways an SVG can declare its color, so
the recoloring tests stay stable and self-documenting even as the production
icons in `src/wiser/resources/` change or get removed.

| File | Property exercised |
|------|--------------------|
| `styled_stroke_black.svg` | color declared as `stroke:#000` inside a `<style>` block |
| `default_fill_black.svg`  | **no** color declared — relies on SVG's default black fill |
| `multicolor.svg`          | several distinct `fill` colors (must survive `monochrome=False`) |

The `default_fill_black.svg` case is the important one: a naive "string-replace
`#000`" recoloring would silently miss it (there is no `#000` to replace), so the
test that uses it guards against that regression.

Tests against the *real* `:/icons/...` resources still exist in `test_theme.py`
to verify the compiled Qt resource pipeline; those are deliberately coupled to
production icons. These fixtures cover the recoloring *behavior* instead.
