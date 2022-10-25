index.html : slides.md slides.css
	pandoc -t revealjs -s -o index.html slides.md -V revealjs-url=https://unpkg.com/reveal.js/ --include-in-header=slides.css -V theme=serif 

upload : index.html
	rsync -avz . jsoma:/var/www/jonathansoma.com/current/site/projects/se4ai
