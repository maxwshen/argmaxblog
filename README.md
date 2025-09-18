# argmax(blog)

Blog hosted on Netlify and built with Hugo and PaperMod, using a
custom setup to support KaTeX/MathJax.

# Setup

```bash
git clone --recurse-submodules https://github.com/maxwshen/argmaxblog.git
sudo apt install hugo
```

Serve locally
```bash
hugo server -D
```

Add markdown blog posts in `/content/posts/`.

### How to add posts to blog

- Write in hackmd.io
- Add markdown file in `content/posts/` folder, and add it to git
    - Ensure markdown starts with:
        
        ---
          title: ""
          math: mathjax / katex
          summary: text to show on home page
          date: 2024-01-04
        ---
        
- Replace all instances of _ with \_: some parsing on two _ in a line causes error with math rendering. Similarly, beware the * symbol.
- Replace all instances of \\ (newline, in block equations) with triple \\\
- Shortcodes for color blocks:
    
    Blue box:
    
    ```python
    {{< box info >}}
    Hello there, and have a nice day
    {{< /box >}}
    
    {{< box important ****>}}
    **bold title**
    
    Hello there, and have a nice day
    {{< /box >}}
    ```
    
    Yellow box: **`{{< box important >}}`.** Other options: **`warning` , `tip`.** 
    
- [Not necessary?] Surround math blocks that use `\\` with the `{{< math >}}` shortcode — this is required as \\ is a protected phrase
- Git commit and push