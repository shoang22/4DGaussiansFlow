DATASET ?= background1-4dgs-og
PORT ?= 6021

train:
	python train.py -s data/multipleview/$(DATASET) --port $(PORT) --expname "multipleview/$(DATASET)" --configs arguments/multipleview/default.py 
pose:
	bash multipleviewprogress.sh $(DATASET)
reset-pose:
	rm -rf ./data/multipleview/$(DATASET)/poses_bounds_multipleview.npy \
	&& rm -rf ./colmap_tmp \
	&& rm -rf ./data/multipleview/$(DATASET)/sparse_ \
	&& rm -rf ./data/multipleview/$(DATASET)/points3D_multipleview.ply
reset-port:
	lsof -iTCP:$(PORT) -sTCP:LISTEN -Pn 2>/dev/null | awk 'NR>1 {print $$2}' | xargs -r kill -9
train-bounce:
	python train.py -s data/dnerf/bouncingballs --port $(PORT) --expname "dnerf/bouncingballs" --configs arguments/dnerf/bouncingballs.py 
