SHELL := /bin/bash

.PHONY:	all

all:	clusterization report

.PHONY:	clusterization report

clusterization:	out-2-6-50 out-2-6-inf
	@echo "Are schema similarities different?" ;  \
	if diff out-schema-similarities-*.csv > /dev/null; then  \
		echo "No, they are equal. Continuing..." ;  \
		cp out-schema-similarities-2-6-50.csv schema-similarities.csv; \
	else  \
		echo "Yes, they are different. Aborting..." ;  \
		false;  \
	fi

report:	clusterization clusterization-report.pdf

.PHONY:	out-2-6-50 out-2-6-inf

out-2-6-50:	out-clusters-2-6-50.csv
out-2-6-inf:	out-clusters-2-6-inf.csv


out-clusters-2-6-50.csv:	entities-similarities.csv ../schema-integration.py
	@echo "Executing schema clusterization with 2 <= n_s <= 6 and n_e <= 50..." ; \
	nice ../schema-integration.py -v -l 2 -u 6 -e 50 \
		-s out-schema-similarities-2-6-50.csv   \
		-g out-schema-similarities-2-6-50.dot   \
		entities-similarities.csv  \
		out-clusters-2-6-50.csv |& tee -i out-messages-2-6-50.log ;  \
	xz -9vf out-messages-2-6-50.log ; \
	rm -f gurobi*


out-clusters-2-6-inf.csv:	entities-similarities.csv ../schema-integration.py
	@echo "Executing schema clusterization with 2 <= n_s <= 6 and n_e unlimited..." ; \
	nice ../schema-integration.py -v -l 2 -u 6 -e 1000 \
		-s out-schema-similarities-2-6-inf.csv    \
		-g out-schema-similarities-2-6-inf.dot    \
		entities-similarities.csv   \
		out-clusters-2-6-inf.csv    \
			|& tee -i out-messages-2-6-inf.log ;  \
	xz -9vf out-messages-2-6-inf.log ; \
	rm -f gurobi*


clusterization-report.pdf:	analysis/clusterization-report.Rnw clusterizations-to-analyze.csv manual-clusters-2-6-50.csv manual-clusters-2-6-inf.csv out-clusters-2-6-50.csv out-clusters-2-6-inf.csv schema-similarities.csv
	@echo "Preparing clusterization report..."; \
	echo 'library("knitr"); opts_knit$$set(root.dir = getwd()); knit2pdf("analysis/clusterization-report.Rnw")' | R --no-save --no-restore ; \
	rm -f clusterization-report.aux clusterization-report.log clusterization-report.out clusterization-report.tex

