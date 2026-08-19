---
name: cite-verify
description: Use when the user asks to find, verify, or update citations in a paper — searches arXiv, ChemRxiv, bioRxiv, and PubMed for real papers matching a topic or claim, verifies author names and titles exist, and formats citations accurately.
---

# Cite-Verify: Find and Verify Real Citations

Follow these steps every time this skill activates.

## Step 1 — Identify what needs citations

Read the target file (usually `paper/summary.md`) with `read_file`.
List every `[N]` reference placeholder or every claim that needs a citation.
Note the topic each citation must cover (e.g. "quantum kernel methods for molecular property prediction").

## Step 2 — Search for real papers

For each topic, run searches using `execute_command` with `curl` against public APIs:

### arXiv
```bash
curl -s "https://export.arxiv.org/api/query?search_query=all:<TERMS>&max_results=5&sortBy=relevance" \
  | grep -E "<title>|<author>|<published>|<id>" | head -40
```

### Semantic Scholar (blocked on this cluster — skip if no response)
```bash
curl --max-time 10 -s "https://api.semanticscholar.org/graph/v1/paper/search?query=<TERMS>&limit=5&fields=title,authors,year,venue,externalIds" \
  | python3 -m json.tool 2>/dev/null | grep -E '"title"|"name"|"year"|"venue"' | head -30
```

Replace `<TERMS>` with URL-encoded search terms (spaces → `+`).

**Note:** On this cluster, only `export.arxiv.org` is reliably reachable. Prefer direct
arXiv ID lookups when you know the paper. Use the keyword search as a fallback.

**Direct lookup by arXiv ID (preferred when ID is known):**
```bash
curl --max-time 15 -s "https://export.arxiv.org/api/query?id_list=XXXX.XXXXX" \
  | python3 -c "
import sys,re; data=sys.stdin.read(); entries=data.split('<entry>')
for e in entries[1:]:
    title=re.search(r'<title>(.*?)</title>',e,re.S)
    authors=re.findall(r'<name>(.*?)</name>',e)
    pub=re.search(r'<published>(.*?)</published>',e)
    doi=re.search(r'<arxiv:doi[^>]*>(.*?)</arxiv:doi>',e,re.S)
    journal=re.search(r'<arxiv:journal_ref[^>]*>(.*?)</arxiv:journal_ref>',e,re.S)
    if title:
        print('TITLE:',title.group(1).strip())
        print('AUTHORS:','; '.join(authors[:4]))
        print('YEAR:',pub.group(1)[:4] if pub else 'N/A')
        print('JOURNAL:',journal.group(1).strip() if journal else 'preprint')
        print('DOI:',doi.group(1).strip() if doi else 'none')
"
```

## Step 3 — Verify each candidate

For each candidate paper returned by the searches:
1. Confirm the **title** matches the intended claim.
2. Note the **first author's last name** and at least two co-authors.
3. Note the **year** and **venue** (journal, conference, or preprint server).
4. If a DOI or arXiv ID is available, record it as a URL.

**Do not cite a paper if you cannot confirm title + author + year from the API response.**
If a search returns no usable result, mark that citation as `[UNVERIFIED — needs manual check]`
and leave a comment in the paper.

## Step 4 — Format citations

Use this format consistently:
```
N. Last, F. M., Co-Author, A. B., & Third, C. D. (YEAR). "Title of the paper."
   *Venue Name* vol(issue), pages. https://doi.org/... or https://arxiv.org/abs/...
```

For preprints with no volume/pages:
```
N. Last, F. M. et al. (YEAR). "Title." *arXiv* / *ChemRxiv* / *bioRxiv* preprint.
   https://arxiv.org/abs/XXXX.XXXXX
```

## Step 5 — Update the paper

Use `apply_diff` (not `write_file`) to replace only the References section of the paper,
inserting the verified citations in the correct format.

Report to the user:
- How many citations were verified (with source)
- How many remain unverified (with the search terms used)
