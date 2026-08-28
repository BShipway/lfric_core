##############################################################################
# Copyright (c) 2017,  Met Office, on behalf of HMSO and Queen's Printer
# For further details please refer to the file LICENCE which you
# should have received as part of this distribution.
##############################################################################
#
# Run this make file to copy a source tree from SOURCE_DIR to WORKING_DIR
#
.PHONY: files-to-extract
files-to-extract: $(addprefix $(WORKING_DIR)/,$(shell find $(SOURCE_DIR) \( -name '*.[Ff]90' -o -name '*.h' -o -name '*.cpp' \) -print | sed "s|$(SOURCE_DIR)/||")) \
                  | $(WORKING_DIR)

.PRECIOUS: $(WORKING_DIR)/%.F90
$(WORKING_DIR)/%.F90: $(SOURCE_DIR)/%.F90 | $(WORKING_DIR)
	$(call MESSAGE,Copying source,$<)
	$(Q)mkdir -p $(dir $@)
	$(Q)cp $< $@

.PRECIOUS: $(WORKING_DIR)/%.f90
$(WORKING_DIR)/%.f90: $(SOURCE_DIR)/%.f90 | $(WORKING_DIR)
	$(call MESSAGE,Copying source,$<)
	$(Q)mkdir -p $(dir $@)
	$(Q)cp $< $@

# Hand-written C++. Generated C++ needs no rule here because PSyclone writes it
# straight into the working directory; this is for C++ that lives in a
# component source tree, which compile.mk then finds alongside it.
#
.PRECIOUS: $(WORKING_DIR)/%.cpp
$(WORKING_DIR)/%.cpp: $(SOURCE_DIR)/%.cpp | $(WORKING_DIR)
	$(call MESSAGE,Copying source,$<)
	$(Q)mkdir -p $(dir $@)
	$(Q)cp $< $@

.PRECIOUS: $(WORKING_DIR)/%.nld
$(WORKING_DIR)/%.nld: $(SOURCE_DIR)/%.nld | $(WORKING_DIR)
	$(call MESSAGE,Copying source,$<)
	$(Q)mkdir -p $(dir $@)
	$(Q)cp $< $@

.PRECIOUS: $(WORKING_DIR)/%.h
$(WORKING_DIR)/%.h: $(SOURCE_DIR)/%.h | $(WORKING_DIR)
	$(call MESSAGE,Copying source,$<)
	$(Q)mkdir -p $(dir $@)
	$(Q)cp $< $@

$(WORKING_DIR):
	$(call MESSAGE,Creating,$@)
	$(Q)mkdir -p $@

include $(LFRIC_BUILD)/lfric.mk
