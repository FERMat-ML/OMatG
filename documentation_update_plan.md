# OMatG Documentation Improvement Plan

## Executive Summary

This plan addresses navigation inconsistencies, mismatched content organization, and structural improvements needed in the OMatG Sphinx documentation. The documentation is well-structured overall but has specific issues with grid card counts vs. actual pages, disabled navigation features, and incomplete cross-referencing.

---

## Issues Identified

### 1. Navigation Problems

#### 1.1 Disabled Sequential Navigation
- **Issue**: `conf.py` has `"show_prev_next": False` in theme options
- **Impact**: Users cannot navigate sequentially through related pages
- **Location**: `/home/gpwolfe/OMatG/docs/source/conf.py`
- **Fix**: Enable prev/next navigation for linear reading flow

#### 1.2 Missing Breadcrumb Context
- **Issue**: No clear "up" or "back to section index" links on individual pages
- **Impact**: Users on deep pages (e.g., API generated docs) lack easy return path
- **Fix**: Add explicit breadcrumb-style navigation or section headers with links

### 2. Mismatched Grid Cards vs. Toctree Entries

#### 2.1 User Guide Index Mismatch
- **Issue**: Grid shows 6 cards but toctree has 8 pages
- **Location**: `/home/gpwolfe/OMatG/docs/source/user_guide/index.rst`
- **Missing from grid**:
  - Training page (`training.rst`)
  - Generation page (`generation.rst`)
- **Impact**: Users viewing the visual grid won't know these pages exist
- **Fix**: Add cards for Training and Generation, update grid layout to `2 2 2 2`

#### 2.2 API Reference Index Mismatch
- **Issue**: Grid shows 5 cards but toctree has 6 modules
- **Location**: `/home/gpwolfe/OMatG/docs/source/api/index.rst`
- **Missing from grid**: Analysis module card
- **Impact**: Analysis module is less discoverable
- **Fix**: Add Analysis card, update grid layout to `2 2 2` or `3 3`

### 3. Cross-Reference Gaps

#### 3.1 Incomplete "See Also" Sections
- **Issue**: User guide pages don't consistently link to corresponding API docs
- **Affected files**:
  - `/home/gpwolfe/OMatG/docs/source/user_guide/stochastic_interpolants.rst`
  - `/home/gpwolfe/OMatG/docs/source/user_guide/sampler.rst`
  - `/home/gpwolfe/OMatG/docs/source/user_guide/datamodule.rst`
  - `/home/gpwolfe/OMatG/docs/source/user_guide/model.rst`
  - `/home/gpwolfe/OMatG/docs/source/user_guide/training.rst`
  - `/home/gpwolfe/OMatG/docs/source/user_guide/generation.rst`
- **Fix**: Add "See Also" sections linking to related API reference pages

#### 3.2 Missing Links to Contributing Guide
- **Issue**: No links from user guide or API pages to contributing guidelines
- **Impact**: Potential contributors don't know how to help
- **Fix**: Add footer or sidebar link to contributing guide

### 4. Content Organization Issues

#### 4.1 Content Duplication
- **Issue**: Training and generation information appears in both:
  - User guide pages (`user_guide/training.rst`, `user_guide/generation.rst`)
  - API reference (`api/training.rst`)
- **Impact**: Confusion about which is canonical, maintenance burden
- **Fix**: Clarify distinction (user guide = how-to, API = reference)

#### 4.2 Index Page Content Gaps
- **Issue**: Section index pages don't provide context about navigation structure
- **Fix**: Add introductory paragraphs explaining page organization

### 5. Naming and Terminology Consistency

#### 5.1 Module Name vs. Display Name
- **Issue**: Inconsistency between module names (`omg.si`) and display names ("Stochastic Interpolants")
- **Impact**: Confusion when searching or cross-referencing
- **Fix**: Use consistent format: "Display Name (`omg.module`)" throughout

#### 5.2 File Naming in Generated Docs
- **Issue**: Auto-generated files use full paths (e.g., `omg.si.interpolants.LinearInterpolant.rst`)
- **Impact**: Not actually a problem - this is Sphinx convention
- **Action**: Document/accept as standard practice

---

## Improvement Tasks

### Phase 1: Critical Navigation Fixes (High Priority)

#### Task 1.1: Fix User Guide Index Grid
- **File**: `/home/gpwolfe/OMatG/docs/source/user_guide/index.rst`
- **Action**:
  1. Add grid card for "Training" page
  2. Add grid card for "Generation" page
  3. Update grid directive to `2 2 2 2` layout
  4. Ensure card titles, icons, and descriptions match page content
- **Expected outcome**: All 8 user guide pages visible in grid

#### Task 1.2: Fix API Reference Index Grid
- **File**: `/home/gpwolfe/OMatG/docs/source/api/index.rst`
- **Action**:
  1. Add grid card for "Analysis" module
  2. Update grid directive to `2 2 2` or `3 3` layout
  3. Ensure card descriptions match module purpose
- **Expected outcome**: All 6 API modules visible in grid

#### Task 1.3: Enable Sequential Navigation
- **File**: `/home/gpwolfe/OMatG/docs/source/conf.py`
- **Action**: Change `"show_prev_next": False` to `"show_prev_next": True`
- **Expected outcome**: Prev/Next links appear at bottom of pages

### Phase 2: Cross-Reference Enhancement (Medium Priority)

#### Task 2.1: Add "See Also" Sections to User Guide Pages
For each user guide page, add a "See Also" section at the end:

- **User Guide → API Mappings**:
  - `stochastic_interpolants.rst` → `:doc:`../api/si``
  - `sampler.rst` → `:doc:`../api/sampler``
  - `datamodule.rst` → `:doc:`../api/datamodule``
  - `model.rst` → `:doc:`../api/model``
  - `analysis.rst` → `:doc:`../api/analysis``
  - `training.rst` → `:doc:`../api/training``
  - `generation.rst` → `:doc:`../api/training`` (generation functions in training module)

- **Template for "See Also" section**:
```rst
See Also
--------
* :doc:`../api/[module]` - API reference for [Module Name]
* :doc:`contributing <../development/contributing>` - How to contribute
```

#### Task 2.2: Add Reciprocal Links from API to User Guide
- **Action**: Verify each API page has "See Also" linking back to user guide
- **Current state**: Some API pages already have this
- **Fix**: Ensure consistency across all API pages

#### Task 2.3: Add Contributing Link to Main Navigation
- **File**: `/home/gpwolfe/OMatG/docs/source/conf.py`
- **Action**: Add "Contributing" to header links in `html_theme_options`
- **Example**:
```python
html_theme_options = {
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "header_links_before_dropdown": 5,
    "external_links": [
        {"name": "Contributing", "url": "development/contributing.html"}
    ],
}
```

### Phase 3: Content Clarity (Medium Priority)

#### Task 3.1: Add Section Context to Index Pages
Add introductory paragraphs to index pages explaining organization:

- **Files to update**:
  - `/home/gpwolfe/OMatG/docs/source/getting_started/index.rst`
  - `/home/gpwolfe/OMatG/docs/source/user_guide/index.rst`
  - `/home/gpwolfe/OMatG/docs/source/api/index.rst`
  - `/home/gpwolfe/OMatG/docs/source/development/index.rst`

- **Content to add**:
  - Brief section purpose
  - Reading order recommendation
  - What to expect in each subsection

#### Task 3.2: Clarify User Guide vs. API Purpose
- **Location**: Top of each user guide page and API page
- **Action**: Add admonition or note explaining:
  - User Guide = Conceptual how-to with examples
  - API Reference = Technical specification and class details

Example:
```rst
.. note::
   This is a conceptual guide. For detailed API documentation, see :doc:`../api/training`.
```

#### Task 3.3: Add Navigation Hints to Deep Pages
- **Target**: Auto-generated API class pages in `/api/generated/`
- **Action**: Add template header with "← Back to [Module] API" link
- **File to modify**: `/home/gpwolfe/OMatG/docs/source/_templates/autosummary/class.rst` (may need to create)

### Phase 4: Naming Consistency (Low Priority)

#### Task 4.1: Standardize Module Display Names
- **Action**: Use consistent format throughout: "Display Name (`omg.module`)"
- **Example**: "Stochastic Interpolants (`omg.si`)" instead of just "Stochastic Interpolants"
- **Files to update**: All index pages and cross-references

#### Task 4.2: Create Glossary Page
- **File**: Create `/home/gpwolfe/OMatG/docs/source/glossary.rst`
- **Content**:
  - Define OMatG-specific terms
  - Map display names to module names
  - Link common abbreviations (CSP, DNG, SI)
- **Add to main index**: Link from main page or sidebar

### Phase 5: Enhanced Navigation Features (Low Priority)

#### Task 5.1: Add Search Configuration
- **File**: `/home/gpwolfe/OMatG/docs/source/conf.py`
- **Action**: Configure search to prioritize certain pages
- **Example**:
```python
html_theme_options = {
    "search_bar_position": "navbar",
}
```

#### Task 5.2: Add Custom CSS for Breadcrumbs
- **File**: Create `/home/gpwolfe/OMatG/docs/source/_static/custom.css`
- **Action**: Style breadcrumb navigation
- **Register in conf.py**:
```python
html_static_path = ['_static']
html_css_files = ['custom.css']
```

#### Task 5.3: Add "Back to Top" Links
- **Action**: Add directive at end of long pages
- **Example**:
```rst
`Back to top <#>`_
```

---

## Verification Checklist

After implementing fixes, verify:

### Navigation Testing
- [ ] All grid cards on index pages match toctree entries
- [ ] Prev/Next links appear and work correctly
- [ ] Links from user guide to API work
- [ ] Links from API back to user guide work
- [ ] Contributing page is easily discoverable
- [ ] No broken internal links (run `make linkcheck`)

### Content Testing
- [ ] Each section index has clear introductory content
- [ ] "See Also" sections are present and accurate
- [ ] No duplicate or conflicting information between user guide and API
- [ ] Naming is consistent throughout (module names, class names)

### User Journey Testing
- [ ] New user can follow Getting Started → User Guide → API flow
- [ ] Developer can find contributing guidelines easily
- [ ] API reference user can navigate back to conceptual docs
- [ ] Generated API class pages have clear return path

### Build Testing
- [ ] `make html` completes without warnings
- [ ] `make linkcheck` shows no broken links
- [ ] All pages render correctly in browser
- [ ] Search functionality works

---

## Implementation Order

### Week 1: Critical Fixes
1. Task 1.1 - Fix User Guide grid (30 min)
2. Task 1.2 - Fix API Reference grid (20 min)
3. Task 1.3 - Enable prev/next navigation (5 min)
4. Build and test (15 min)

### Week 2: Cross-References
1. Task 2.1 - Add "See Also" to all user guide pages (2 hours)
2. Task 2.2 - Verify API reciprocal links (30 min)
3. Task 2.3 - Add contributing to navigation (30 min)
4. Build and test (15 min)

### Week 3: Content Improvements
1. Task 3.1 - Add context to index pages (1 hour)
2. Task 3.2 - Clarify user guide vs API (1 hour)
3. Task 3.3 - Navigation hints on deep pages (1 hour)
4. Build and test (15 min)

### Week 4: Polish (Optional)
1. Task 4.1 - Standardize naming (1 hour)
2. Task 4.2 - Create glossary (2 hours)
3. Tasks 5.1-5.3 - Enhanced navigation (2 hours)
4. Final build and comprehensive test (30 min)

---

## Success Metrics

- **Navigation**: Zero dead ends without clear return path
- **Consistency**: 100% match between grid cards and toctree entries
- **Discovery**: All major pages reachable within 3 clicks from main index
- **Links**: Zero broken internal links (verified with `make linkcheck`)
- **User feedback**: Reduced confusion about documentation structure

---

## Files Reference

### Critical Files to Modify
```
/home/gpwolfe/OMatG/docs/source/conf.py
/home/gpwolfe/OMatG/docs/source/user_guide/index.rst
/home/gpwolfe/OMatG/docs/source/api/index.rst
```

### User Guide Pages (for cross-references)
```
/home/gpwolfe/OMatG/docs/source/user_guide/stochastic_interpolants.rst
/home/gpwolfe/OMatG/docs/source/user_guide/sampler.rst
/home/gpwolfe/OMatG/docs/source/user_guide/datamodule.rst
/home/gpwolfe/OMatG/docs/source/user_guide/model.rst
/home/gpwolfe/OMatG/docs/source/user_guide/analysis.rst
/home/gpwolfe/OMatG/docs/source/user_guide/training.rst
/home/gpwolfe/OMatG/docs/source/user_guide/generation.rst
```

### API Pages (for verification)
```
/home/gpwolfe/OMatG/docs/source/api/si.rst
/home/gpwolfe/OMatG/docs/source/api/sampler.rst
/home/gpwolfe/OMatG/docs/source/api/datamodule.rst
/home/gpwolfe/OMatG/docs/source/api/model.rst
/home/gpwolfe/OMatG/docs/source/api/analysis.rst
/home/gpwolfe/OMatG/docs/source/api/training.rst
```

---

## Notes

- Phase 1 tasks are essential and should be completed first
- Phases 2-3 significantly improve usability
- Phase 4-5 tasks are nice-to-have enhancements
- All changes should be tested with `make html` and `make linkcheck`
- Consider user feedback after Phase 1-2 before proceeding to later phases
