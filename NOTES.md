# CV Project
- Overleaf link: https://cs.overleaf.com/8893629753dnjfpjwvpfhq
- Dataset links:
	- Smíchoff (Czech Republic):
		- raw: https://tom.ggu.cz/cv-sm-data/
		- downsized: https://tom.ggu.cz/cv-sm-data-downsized/
	- Boulderhaus (Germany):
		- raw: https://tom.ggu.cz/cv-bh-data/
		- downsized: https://tom.ggu.cz/cv-bh-data-downsized/

## Data Annotation
- https://labelstud.io/
- https://www.robots.ox.ac.uk/~vgg/software/via/


## Related Papers/Repos
- https://kastner.ucsd.edu/ryan/wp-content/uploads/sites/5/2022/06/admin/rock-climbing-coach.pdf
- https://stacks.stanford.edu/file/druid:bf950qp8995/Wei.pdf
- https://github.com/cydivision/climbnet
- https://cs229.stanford.edu/proj2017/final-reports/5232206.pdf


## Teams
**Learning Based:** Shrey, Philipp
**CV Based:** Tom, Kiryl


## Roadmap
-  9.3.: initial dataset (manual annotation, 5-10 images)
- 16.3.: sync meeting
- 23.3.: sync meeting, start writing the paper
- 30.3.: deadline


## Task 1: hold recognition
- **input:** 2D picture of a climbing wall
    - images are rectangular, we can do pre/post processing if they're not
- **output:** polygon bounding box of the holds on the wall


## Task 2: route recognition
- **input:** 2D picture of a climbing wall + locations of holds
- **output:** clusters of holds that form a route