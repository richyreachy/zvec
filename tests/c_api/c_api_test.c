// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "zvec/c_api.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <_printf.h>
#ifdef _POSIX_C_SOURCE
#include <sys/time.h>
#endif
#include <unistd.h>
#include "utils.h"

// =============================================================================
// Test helper macro definitions
// =============================================================================

static int test_count = 0;
static int passed_count = 0;
static int current_test_passed = 1;  // Track if current test function passes

#define TEST_START()                        \
  do {                                      \
    printf("Running test: %s\n", __func__); \
    test_count++;                           \
    current_test_passed = 1;                \
  } while (0)

#define TEST_ASSERT(condition)                   \
  do {                                           \
    if (condition) {                             \
      printf("  ✓ PASS\n");                      \
    } else {                                     \
      printf("  ✗ FAIL at line %d\n", __LINE__); \
      current_test_passed = 0;                   \
    }                                            \
  } while (0)

#define TEST_END()             \
  do {                         \
    if (current_test_passed) { \
      passed_count++;          \
    }                          \
  } while (0)

// =============================================================================
// Helper functions tests
// =============================================================================

void test_version_functions(void) {
  TEST_START();

  // Test version retrieval functions
  const char *version = zvec_get_version();
  TEST_ASSERT(version != NULL);

  // Test version component retrieval
  int major = zvec_get_version_major();
  int minor = zvec_get_version_minor();
  int patch = zvec_get_version_patch();

  TEST_ASSERT(major >= 0);
  TEST_ASSERT(minor >= 0);
  TEST_ASSERT(patch >= 0);

  TEST_ASSERT(zvec_check_version(major, minor, patch));

  // Test version checking functions
  bool compatible = zvec_check_version(0, 3, 0);
  TEST_ASSERT(compatible == true);

  bool not_compatible = zvec_check_version(99, 99, 99);
  TEST_ASSERT(not_compatible == false);

  TEST_END();
}

void test_error_handling_functions(void) {
  TEST_START();

  char *error_msg = NULL;
  ZVecErrorCode err = zvec_get_last_error(&error_msg);
  TEST_ASSERT(err == ZVEC_OK);

  if (error_msg) {
    zvec_free_str(error_msg);
  }

  // Test error clearing
  zvec_clear_error();

  // Test error details retrieval
  ZVecErrorDetails error_details = {0};
  err = zvec_get_last_error_details(&error_details);
  TEST_ASSERT(err == ZVEC_OK);

  TEST_END();
}

void test_zvec_config() {
  TEST_START();

  // Test 1: Console log config creation and destruction
  ZVecConsoleLogConfig *console_config =
      zvec_config_console_log_create(ZVEC_LOG_LEVEL_INFO);
  TEST_ASSERT(console_config != NULL);
  if (console_config) {
    TEST_ASSERT(console_config->level == ZVEC_LOG_LEVEL_INFO);
    zvec_config_console_log_destroy(console_config);
  }

  // Test 2: File log config creation and destruction
  ZVecFileLogConfig *file_config = zvec_config_file_log_create(
      ZVEC_LOG_LEVEL_WARN, "./logs", "test_log", 100, 7);
  TEST_ASSERT(file_config != NULL);
  if (file_config) {
    TEST_ASSERT(file_config->level == ZVEC_LOG_LEVEL_WARN);
    TEST_ASSERT(strcmp(file_config->dir.data, "./logs") == 0);
    TEST_ASSERT(strcmp(file_config->basename.data, "test_log") == 0);
    TEST_ASSERT(file_config->file_size == 100);
    TEST_ASSERT(file_config->overdue_days == 7);
    zvec_config_file_log_destroy(file_config);
  }

  // Test 3: File log config edge cases
  ZVecFileLogConfig *empty_file_config =
      zvec_config_file_log_create(ZVEC_LOG_LEVEL_INFO, "", "", 0, 0);
  TEST_ASSERT(empty_file_config != NULL);
  if (empty_file_config) {
    TEST_ASSERT(empty_file_config->level == ZVEC_LOG_LEVEL_INFO);
    TEST_ASSERT(strcmp(empty_file_config->dir.data, "") == 0);
    TEST_ASSERT(strcmp(empty_file_config->basename.data, "") == 0);
    TEST_ASSERT(empty_file_config->file_size == 0);
    TEST_ASSERT(empty_file_config->overdue_days == 0);
    zvec_config_file_log_destroy(empty_file_config);
  }

  // Test 4: Log config creation with console type
  ZVecConsoleLogConfig *temp_console =
      zvec_config_console_log_create(ZVEC_LOG_LEVEL_ERROR);
  ZVecLogConfig *log_config_console =
      zvec_config_log_create(ZVEC_LOG_TYPE_CONSOLE, temp_console);
  TEST_ASSERT(log_config_console != NULL);
  if (log_config_console) {
    TEST_ASSERT(log_config_console->type == ZVEC_LOG_TYPE_CONSOLE);
    TEST_ASSERT(log_config_console->config.console_config.level ==
                ZVEC_LOG_LEVEL_ERROR);
    zvec_config_log_destroy(log_config_console);
  }
  if (temp_console) {
    zvec_config_console_log_destroy(temp_console);
  }

  // Test 5: Log config creation with file type
  ZVecFileLogConfig *temp_file = zvec_config_file_log_create(
      ZVEC_LOG_LEVEL_DEBUG, "./logs", "app", 50, 30);
  ZVecLogConfig *log_config_file =
      zvec_config_log_create(ZVEC_LOG_TYPE_FILE, temp_file);
  TEST_ASSERT(log_config_file != NULL);
  if (log_config_file) {
    TEST_ASSERT(log_config_file->type == ZVEC_LOG_TYPE_FILE);
    TEST_ASSERT(log_config_file->config.file_config.level ==
                ZVEC_LOG_LEVEL_DEBUG);
    TEST_ASSERT(
        strcmp(log_config_file->config.file_config.dir.data, "./logs") == 0);
    TEST_ASSERT(
        strcmp(log_config_file->config.file_config.basename.data, "app") == 0);
    zvec_config_log_destroy(log_config_file);
  }
  if (temp_file) {
    zvec_config_file_log_destroy(temp_file);
  }

  // Test 6: Log config with NULL config data (should use defaults)
  ZVecLogConfig *log_config_default =
      zvec_config_log_create(ZVEC_LOG_TYPE_CONSOLE, NULL);
  TEST_ASSERT(log_config_default != NULL);
  if (log_config_default) {
    TEST_ASSERT(log_config_default->type == ZVEC_LOG_TYPE_CONSOLE);
    TEST_ASSERT(log_config_default->config.console_config.level ==
                ZVEC_LOG_LEVEL_WARN);
    zvec_config_log_destroy(log_config_default);
  }

  // Test 7: Config data creation and basic operations
  ZVecConfigData *config_data = zvec_config_data_create();
  TEST_ASSERT(config_data != NULL);
  if (config_data) {
    // Test initial values
    TEST_ASSERT(config_data->log_config != NULL);
    TEST_ASSERT(config_data->log_config->type == ZVEC_LOG_TYPE_CONSOLE);

    // Test memory limit setting
    ZVecErrorCode err =
        zvec_config_data_set_memory_limit(config_data, 1024 * 1024 * 1024);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(config_data->memory_limit_bytes == 1024 * 1024 * 1024);

    // Test thread count settings
    err = zvec_config_data_set_query_thread_count(config_data, 8);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(config_data->query_thread_count == 8);

    err = zvec_config_data_set_optimize_thread_count(config_data, 4);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(config_data->optimize_thread_count == 4);

    // Test log config replacement
    ZVecConsoleLogConfig *new_console =
        zvec_config_console_log_create(ZVEC_LOG_LEVEL_DEBUG);
    ZVecLogConfig *new_log_config =
        zvec_config_log_create(ZVEC_LOG_TYPE_CONSOLE, new_console);
    if (new_log_config) {
      err = zvec_config_data_set_log_config(config_data, new_log_config);
      TEST_ASSERT(err == ZVEC_OK);
      TEST_ASSERT(config_data->log_config == new_log_config);
    }

    zvec_config_data_destroy(config_data);
    if (new_console) zvec_config_console_log_destroy(new_console);
    if (new_log_config) zvec_config_log_destroy(new_log_config);
  }

  // Test 8: Edge cases and error conditions
  // Test NULL pointer handling
  ZVecErrorCode err = zvec_config_data_set_memory_limit(NULL, 1024);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  err = zvec_config_data_set_log_config(NULL, NULL);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  err = zvec_config_data_set_query_thread_count(NULL, 1);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  err = zvec_config_data_set_optimize_thread_count(NULL, 1);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  // Test boundary values
  ZVecConfigData *boundary_config = zvec_config_data_create();
  if (boundary_config) {
    // Test zero values
    err = zvec_config_data_set_memory_limit(boundary_config, 0);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(boundary_config->memory_limit_bytes == 0);

    // Test maximum values
    err = zvec_config_data_set_memory_limit(boundary_config, UINT64_MAX);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(boundary_config->memory_limit_bytes == UINT64_MAX);

    // Test zero thread counts
    err = zvec_config_data_set_query_thread_count(boundary_config, 0);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(boundary_config->query_thread_count == 0);

    err = zvec_config_data_set_optimize_thread_count(boundary_config, 0);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(boundary_config->optimize_thread_count == 0);

    zvec_config_data_destroy(boundary_config);
  }

  // Test 9: Memory leak prevention - double destroy safety
  ZVecConfigData *double_destroy_test = zvec_config_data_create();
  if (double_destroy_test) {
    zvec_config_data_destroy(double_destroy_test);
  }

  TEST_END();
}

void test_zvec_initialize() {
  TEST_START();

  ZVecConfigData *config = zvec_config_data_create();
  TEST_ASSERT(config != NULL);
  if (config) {
    TEST_ASSERT(config->log_config != NULL);
    TEST_ASSERT(config->log_config->type == ZVEC_LOG_TYPE_CONSOLE);
  }
  ZVecErrorCode err = zvec_initialize(config);
  TEST_ASSERT(err == ZVEC_OK);
  bool is_initialized = false;
  zvec_is_initialized(&is_initialized);
  TEST_ASSERT(is_initialized);

  TEST_END();
}

// =============================================================================
// Schema-related tests
// =============================================================================

void test_schema_basic_operations(void) {
  TEST_START();

  // Test 1: Basic Schema creation and destruction
  ZVecCollectionSchema *schema = zvec_collection_schema_create("demo");
  TEST_ASSERT(schema != NULL);
  TEST_ASSERT(schema->name != NULL);
  TEST_ASSERT(strcmp(schema->name->data, "demo") == 0);
  TEST_ASSERT(schema->field_count == 0);
  TEST_ASSERT(schema->fields == NULL);
  TEST_ASSERT(schema->max_doc_count_per_segment > 0);

  // Test 2: Schema field count operations
  size_t initial_count = zvec_collection_schema_get_field_count(schema);
  TEST_ASSERT(initial_count == 0);

  // Test 3: Adding fields to schema
  ZVecFieldSchema *id_field =
      zvec_field_schema_create("id", ZVEC_DATA_TYPE_INT64, false, 0);
  ZVecErrorCode err = zvec_collection_schema_add_field(schema, id_field);
  TEST_ASSERT(err == ZVEC_OK);

  size_t count_after_add = zvec_collection_schema_get_field_count(schema);
  TEST_ASSERT(count_after_add == 1);

  // Test 4: Finding fields in schema
  const ZVecFieldSchema *found_field =
      zvec_collection_schema_find_field(schema, "id");
  TEST_ASSERT(found_field != NULL);
  TEST_ASSERT(strcmp(found_field->name->data, "id") == 0);
  TEST_ASSERT(found_field->data_type == ZVEC_DATA_TYPE_INT64);

  // Test 5: Getting field by index
  ZVecFieldSchema *indexed_field = zvec_collection_schema_get_field(schema, 0);
  TEST_ASSERT(indexed_field != NULL);
  TEST_ASSERT(strcmp(indexed_field->name->data, "id") == 0);

  // Test 6: Adding multiple fields
  ZVecFieldSchema fields_to_add[2];
  ZVecFieldSchema *name_field =
      zvec_field_schema_create("name", ZVEC_DATA_TYPE_STRING, false, 0);
  ZVecFieldSchema *age_field =
      zvec_field_schema_create("age", ZVEC_DATA_TYPE_INT32, true, 0);

  fields_to_add[0] = *name_field;
  fields_to_add[1] = *age_field;

  err = zvec_collection_schema_add_fields(schema, fields_to_add, 2);
  TEST_ASSERT(err == ZVEC_OK);

  size_t count_after_multi_add = zvec_collection_schema_get_field_count(schema);
  TEST_ASSERT(count_after_multi_add == 3);

  // Test 7: Finding newly added fields
  const ZVecFieldSchema *name_found =
      zvec_collection_schema_find_field(schema, "name");
  TEST_ASSERT(name_found != NULL);
  TEST_ASSERT(strcmp(name_found->name->data, "name") == 0);

  const ZVecFieldSchema *age_found =
      zvec_collection_schema_find_field(schema, "age");
  TEST_ASSERT(age_found != NULL);
  TEST_ASSERT(strcmp(age_found->name->data, "age") == 0);

  // Test 8: Setting and getting max doc count
  err = zvec_collection_schema_set_max_doc_count_per_segment(schema, 10000);
  TEST_ASSERT(err == ZVEC_OK);

  uint64_t max_doc_count =
      zvec_collection_schema_get_max_doc_count_per_segment(schema);
  TEST_ASSERT(max_doc_count == 10000);

  // Test 9: Schema validation
  ZVecString *validation_error = NULL;
  err = zvec_collection_schema_validate(schema, &validation_error);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(validation_error == NULL);

  // Test 10: Removing single field
  err = zvec_collection_schema_remove_field(schema, "age");
  TEST_ASSERT(err == ZVEC_OK);

  size_t count_after_remove = zvec_collection_schema_get_field_count(schema);
  TEST_ASSERT(count_after_remove == 2);

  const ZVecFieldSchema *removed_field =
      zvec_collection_schema_find_field(schema, "age");
  TEST_ASSERT(removed_field == NULL);

  // Test 11: Removing multiple fields
  const char *fields_to_remove[] = {"name", "id"};
  err = zvec_collection_schema_remove_fields(schema, fields_to_remove, 2);
  TEST_ASSERT(err == ZVEC_OK);

  size_t final_count = zvec_collection_schema_get_field_count(schema);
  TEST_ASSERT(final_count == 0);

  // Test 12: Schema cleanup
  zvec_collection_schema_destroy(schema);

  TEST_END();
}

void test_schema_edge_cases(void) {
  TEST_START();

  // Test 1: NULL parameter handling for schema creation
  ZVecCollectionSchema *null_schema = zvec_collection_schema_create(NULL);
  TEST_ASSERT(null_schema == NULL);

  // Test 2: Empty string schema name
  ZVecCollectionSchema *empty_schema = zvec_collection_schema_create("");
  TEST_ASSERT(empty_schema != NULL);
  TEST_ASSERT(empty_schema->name != NULL);
  TEST_ASSERT(strcmp(empty_schema->name->data, "") == 0);
  zvec_collection_schema_destroy(empty_schema);

  // Test 3: Very long schema name
  char long_name[1024];
  memset(long_name, 'a', 1023);
  long_name[1023] = '\0';
  ZVecCollectionSchema *long_schema = zvec_collection_schema_create(long_name);
  TEST_ASSERT(long_schema != NULL);
  TEST_ASSERT(long_schema->name != NULL);
  TEST_ASSERT(strlen(long_schema->name->data) == 1023);
  zvec_collection_schema_destroy(long_schema);

  // Test 4: NULL schema parameter handling for all functions
  ZVecErrorCode err;
  size_t count = zvec_collection_schema_get_field_count(NULL);
  TEST_ASSERT(count == 0);

  const ZVecFieldSchema *null_field =
      zvec_collection_schema_find_field(NULL, "test");
  TEST_ASSERT(null_field == NULL);

  ZVecFieldSchema *null_indexed_field =
      zvec_collection_schema_get_field(NULL, 0);
  TEST_ASSERT(null_indexed_field == NULL);

  uint64_t null_max_doc_count =
      zvec_collection_schema_get_max_doc_count_per_segment(NULL);
  TEST_ASSERT(null_max_doc_count == 0);

  err = zvec_collection_schema_set_max_doc_count_per_segment(NULL, 1000);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  ZVecString *null_validation_error = NULL;
  err = zvec_collection_schema_validate(NULL, &null_validation_error);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);
  TEST_ASSERT(null_validation_error == NULL);

  err = zvec_collection_schema_add_field(NULL, NULL);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  err = zvec_collection_schema_add_fields(NULL, NULL, 0);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  err = zvec_collection_schema_remove_field(NULL, "test");
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  const char *null_field_names[] = {NULL};
  err = zvec_collection_schema_remove_fields(NULL, null_field_names, 1);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  // Test 5: Working with valid schema for edge cases
  ZVecCollectionSchema *schema = zvec_collection_schema_create("edge_test");
  TEST_ASSERT(schema != NULL);

  // Test 6: Adding NULL field to schema
  err = zvec_collection_schema_add_field(schema, NULL);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  // Test 7: Adding fields with NULL array
  err = zvec_collection_schema_add_fields(schema, NULL, 5);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  // Test 8: Adding zero fields
  err = zvec_collection_schema_add_fields(schema, NULL, 0);
  TEST_ASSERT(err == ZVEC_OK);

  // Test 9: Finding field with NULL name
  const ZVecFieldSchema *null_name_field =
      zvec_collection_schema_find_field(schema, NULL);
  TEST_ASSERT(null_name_field == NULL);

  // Test 10: Finding non-existent field
  const ZVecFieldSchema *nonexistent_field =
      zvec_collection_schema_find_field(schema, "nonexistent");
  TEST_ASSERT(nonexistent_field == NULL);

  // Test 11: Getting field with invalid index
  ZVecFieldSchema *invalid_index_field =
      zvec_collection_schema_get_field(schema, 1000);
  TEST_ASSERT(invalid_index_field == NULL);

  // Test 12: Getting field from empty schema with index 0
  ZVecFieldSchema *zero_index_field =
      zvec_collection_schema_get_field(schema, 0);
  TEST_ASSERT(zero_index_field == NULL);

  // Test 13: Removing field with NULL name
  err = zvec_collection_schema_remove_field(schema, NULL);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  // Test 14: Removing non-existent field
  err = zvec_collection_schema_remove_field(schema, "nonexistent");
  TEST_ASSERT(err == ZVEC_ERROR_NOT_FOUND);

  // Test 15: Removing fields with NULL array
  err = zvec_collection_schema_remove_fields(schema, NULL, 5);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  // Test 16: Removing zero fields
  err = zvec_collection_schema_remove_fields(schema, NULL, 0);
  TEST_ASSERT(err == ZVEC_OK);

  // Test 17: Setting extremely large max doc count
  err =
      zvec_collection_schema_set_max_doc_count_per_segment(schema, UINT64_MAX);
  TEST_ASSERT(err == ZVEC_OK);
  uint64_t retrieved_max_count =
      zvec_collection_schema_get_max_doc_count_per_segment(schema);
  TEST_ASSERT(retrieved_max_count == UINT64_MAX);

  // Test 18: Setting zero max doc count
  err = zvec_collection_schema_set_max_doc_count_per_segment(schema, 0);
  TEST_ASSERT(err == ZVEC_OK);
  uint64_t zero_max_count =
      zvec_collection_schema_get_max_doc_count_per_segment(schema);
  TEST_ASSERT(zero_max_count == 0);

  // Test 19: Schema validation with empty schema
  ZVecString *empty_validation_error = NULL;
  err = zvec_collection_schema_validate(schema, &empty_validation_error);
  TEST_ASSERT(err == ZVEC_ERROR_INVALID_ARGUMENT);

  // Test 20: Add duplicate field names
  ZVecFieldSchema *first_id =
      zvec_field_schema_create("duplicate_id", ZVEC_DATA_TYPE_INT64, false, 0);
  ZVecFieldSchema *second_id =
      zvec_field_schema_create("duplicate_id", ZVEC_DATA_TYPE_STRING, false, 0);

  err = zvec_collection_schema_add_field(schema, first_id);
  TEST_ASSERT(err == ZVEC_OK);

  err = zvec_collection_schema_add_field(schema, second_id);
  TEST_ASSERT(err == ZVEC_ERROR_ALREADY_EXISTS);
  zvec_field_schema_destroy(second_id);

  // Verify fields
  size_t field_count = zvec_collection_schema_get_field_count(schema);
  TEST_ASSERT(field_count == 1);

  // Test 21: Cleanup
  zvec_collection_schema_destroy(schema);

  TEST_END();
}

void test_schema_field_operations(void) {
  TEST_START();

  ZVecCollectionSchema *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    // Test field count
    size_t initial_count = zvec_collection_schema_get_field_count(schema);
    TEST_ASSERT(initial_count == 5);

    // Test finding non-existent field
    const ZVecFieldSchema *nonexistent =
        zvec_collection_schema_find_field(schema, "nonexistent");
    TEST_ASSERT(nonexistent == NULL);

    // Test finding existing field
    const ZVecFieldSchema *id_field =
        zvec_collection_schema_find_field(schema, "id");
    TEST_ASSERT(id_field != NULL);
    if (id_field) {
      TEST_ASSERT(strcmp(id_field->name->data, "id") == 0);
      TEST_ASSERT(id_field->data_type == ZVEC_DATA_TYPE_INT64);
    }

    zvec_collection_schema_destroy(schema);
  }

  TEST_END();
}

void test_normal_schema_creation(void) {
  TEST_START();

  ZVecCollectionSchema *schema =
      zvec_test_create_normal_schema(false, "test_normal", NULL, NULL, 1000);
  TEST_ASSERT(schema != NULL);

  if (schema) {
    TEST_ASSERT(strcmp(schema->name->data, "test_normal") == 0);

    // Verify field count
    size_t field_count = zvec_collection_schema_get_field_count(schema);
    TEST_ASSERT(field_count > 0);

    zvec_collection_schema_destroy(schema);
  }

  TEST_END();
}

void test_schema_with_indexes(void) {
  TEST_START();

  // Test Schema with scalar index
  ZVecCollectionSchema *scalar_index_schema =
      zvec_test_create_schema_with_scalar_index(true, true,
                                                "scalar_index_test");
  TEST_ASSERT(scalar_index_schema != NULL);
  if (scalar_index_schema) {
    zvec_collection_schema_destroy(scalar_index_schema);
  }

  // Test Schema with vector index
  ZVecCollectionSchema *vector_index_schema =
      zvec_test_create_schema_with_vector_index(false, "vector_index_test",
                                                NULL);
  TEST_ASSERT(vector_index_schema != NULL);
  if (vector_index_schema) {
    zvec_collection_schema_destroy(vector_index_schema);
  }

  TEST_END();
}

void test_schema_max_doc_count(void) {
  TEST_START();

  // Test 1: Setting max doc count to a valid value
  ZVecCollectionSchema *schema = zvec_collection_schema_create("max_doc_test");
  TEST_ASSERT(schema != NULL);

  ZVecErrorCode err =
      zvec_collection_schema_set_max_doc_count_per_segment(schema, 1000);
  TEST_ASSERT(err == ZVEC_OK);

  uint64_t max_doc_count =
      zvec_collection_schema_get_max_doc_count_per_segment(schema);
  TEST_ASSERT(max_doc_count == 1000);

  zvec_collection_schema_destroy(schema);

  // Test 2: Setting max doc count to zero
  schema = zvec_collection_schema_create("max_doc_test");
  TEST_ASSERT(schema != NULL);

  err = zvec_collection_schema_set_max_doc_count_per_segment(schema, 0);
  TEST_ASSERT(err == ZVEC_OK);

  max_doc_count = zvec_collection_schema_get_max_doc_count_per_segment(schema);
  TEST_ASSERT(max_doc_count == 0);

  zvec_collection_schema_destroy(schema);

  // Test 3: Setting max doc count to maximum value
  schema = zvec_collection_schema_create("max_doc_test");
  TEST_ASSERT(schema != NULL);

  err =
      zvec_collection_schema_set_max_doc_count_per_segment(schema, UINT64_MAX);
  TEST_ASSERT(err == ZVEC_OK);

  max_doc_count = zvec_collection_schema_get_max_doc_count_per_segment(schema);
  TEST_ASSERT(max_doc_count == UINT64_MAX);

  zvec_collection_schema_destroy(schema);

  TEST_END();
}

// =============================================================================
// Collection-related tests
// =============================================================================

void test_collection_basic_operations(void) {
  TEST_START();

  // Create temporary directory
  char temp_dir[] = "/tmp/zvec_test_collection_basic_operations";

  ZVecCollectionSchema *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    ZVecCollection *collection = NULL;
    ZVecErrorCode err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(collection != NULL);

    if (collection) {
      // Test collection operations
      ZVecDoc *doc1 = zvec_test_create_doc(1, schema, NULL);
      ZVecDoc *doc2 = zvec_test_create_doc(2, schema, NULL);
      ZVecDoc *doc3 = zvec_test_create_doc(3, schema, NULL);

      TEST_ASSERT(doc1 != NULL);
      TEST_ASSERT(doc2 != NULL);
      TEST_ASSERT(doc3 != NULL);

      if (doc1 && doc2 && doc3) {
        ZVecDoc *docs[] = {doc1, doc2, doc3};
        size_t success_count, error_count;

        // Test insert operation
        err = zvec_collection_insert(collection, (const ZVecDoc **)docs, 3,
                                     &success_count, &error_count);
        TEST_ASSERT(err == ZVEC_OK);
        TEST_ASSERT(success_count == 3);
        TEST_ASSERT(error_count == 0);

        // Test update operation
        zvec_doc_set_score(doc1, 0.95f);
        ZVecDoc *update_docs[] = {doc1};
        err = zvec_collection_update(collection, (const ZVecDoc **)update_docs,
                                     1, &success_count, &error_count);
        TEST_ASSERT(err == ZVEC_OK);
        TEST_ASSERT(success_count == 1);
        TEST_ASSERT(error_count == 0);

        // Test upsert operation
        zvec_doc_set_pk(doc3, "pk_3_modified");
        ZVecDoc *upsert_docs[] = {doc3};
        err = zvec_collection_upsert(collection, (const ZVecDoc **)upsert_docs,
                                     1, &success_count, &error_count);
        TEST_ASSERT(err == ZVEC_OK);
        TEST_ASSERT(success_count == 1);
        TEST_ASSERT(error_count == 0);

        // Test delete operation by primary keys
        const char *pks[] = {"pk_1", "pk_2"};
        err = zvec_collection_delete(collection, pks, 2, &success_count,
                                     &error_count);
        TEST_ASSERT(err == ZVEC_OK);
        TEST_ASSERT(success_count == 2);
        TEST_ASSERT(error_count == 0);

        // Test delete by filter
        err = zvec_collection_delete_by_filter(collection, "id > 0");
        TEST_ASSERT(err == ZVEC_OK);

        // Clean up documents
        zvec_doc_destroy(doc1);
        zvec_doc_destroy(doc2);
        zvec_doc_destroy(doc3);
      }

      // Test collection flush
      err = zvec_collection_flush(collection);
      TEST_ASSERT(err == ZVEC_OK);

      // Test collection optimization
      err = zvec_collection_optimize(collection);
      TEST_ASSERT(err == ZVEC_OK);

      zvec_collection_destroy(collection);
    }

    zvec_collection_schema_destroy(schema);
  }

  // Clean up temporary directory
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", temp_dir);
  system(cmd);

  TEST_END();
}

void test_collection_edge_cases(void) {
  TEST_START();

  char temp_dir[] = "/tmp/zvec_test_collection_edge_cases";

  ZVecCollectionSchema *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    ZVecCollection *collection = NULL;

    // Test empty name collection
    ZVecErrorCode err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);
    if (collection) {
      zvec_collection_destroy(collection);
      collection = NULL;
    }

    // Test long name collection
    char long_name[256];
    memset(long_name, 'a', 255);
    long_name[255] = '\0';

    char long_path[512];
    snprintf(long_path, sizeof(long_path), "%s/%s", temp_dir,
             "very_long_collection_name_that_tests_path_limits");

    err = zvec_collection_create_and_open(long_path, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);
    if (collection) {
      zvec_collection_destroy(collection);
      collection = NULL;
    }

    // Test NULL name集合
    err = zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err != ZVEC_OK);

    zvec_collection_schema_destroy(schema);
  }

  // Clean up temporary directory
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", temp_dir);
  system(cmd);

  TEST_END();
}

void test_collection_delete_by_filter(void) {
  TEST_START();

  char temp_dir[] = "/tmp/zvec_test_collection_delete_by_filter";

  ZVecCollectionSchema *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    ZVecCollection *collection = NULL;
    ZVecErrorCode err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);

    if (collection) {
      // Test normal deletion filtering
      err = zvec_collection_delete_by_filter(collection, "id > 1");
      TEST_ASSERT(err == ZVEC_OK);

      // Test NULL filter
      err = zvec_collection_delete_by_filter(collection, NULL);
      TEST_ASSERT(err != ZVEC_OK);

      // Test empty string filter
      err = zvec_collection_delete_by_filter(collection, "");
      TEST_ASSERT(err == ZVEC_OK);

      zvec_collection_destroy(collection);
    }

    zvec_collection_schema_destroy(schema);
  }

  // Clean up temporary directory
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", temp_dir);
  system(cmd);

  TEST_END();
}

void test_collection_stats(void) {
  TEST_START();

  char temp_dir[] = "/tmp/zvec_test_collection_stats";

  ZVecCollectionSchema *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    ZVecCollection *collection = NULL;
    ZVecErrorCode err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);

    if (collection) {
      ZVecCollectionStats *stats = NULL;
      err = zvec_collection_get_stats(collection, &stats);
      TEST_ASSERT(err == ZVEC_OK);

      if (stats) {
        // Basic validation of statistics
        TEST_ASSERT(stats->doc_count ==
                    0);  // New collection should have no documents
        zvec_collection_stats_destroy(stats);
      }

      zvec_collection_destroy(collection);
    }

    zvec_collection_schema_destroy(schema);
  }

  // Clean up temporary directory
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", temp_dir);
  system(cmd);

  TEST_END();
}

// =============================================================================
// Field-related tests
// =============================================================================

void test_field_schema_functions(void) {
  TEST_START();

  // Test scalar field creation
  ZVecFieldSchema scalar_field = {0};
  ZVecString name1 = {0};
  name1.data = "test_field";
  name1.length = 10;
  scalar_field.name = &name1;
  scalar_field.data_type = ZVEC_DATA_TYPE_STRING;
  scalar_field.nullable = true;
  scalar_field.dimension = 0;

  TEST_ASSERT(strcmp(scalar_field.name->data, "test_field") == 0);
  TEST_ASSERT(scalar_field.data_type == ZVEC_DATA_TYPE_STRING);
  TEST_ASSERT(scalar_field.nullable == true);

  // Test vector field creation
  ZVecFieldSchema vector_field = {0};
  ZVecString name2 = {0};
  name2.data = "vec_field";
  name2.length = 9;
  vector_field.name = &name2;
  vector_field.data_type = ZVEC_DATA_TYPE_VECTOR_FP32;
  vector_field.nullable = false;
  vector_field.dimension = 128;

  TEST_ASSERT(strcmp(vector_field.name->data, "vec_field") == 0);
  TEST_ASSERT(vector_field.data_type == ZVEC_DATA_TYPE_VECTOR_FP32);
  TEST_ASSERT(vector_field.dimension == 128);

  // Test sparse vector field creation
  ZVecFieldSchema sparse_field = {0};
  ZVecString name3 = {0};
  name3.data = "sparse_field";
  name3.length = 12;
  sparse_field.name = &name3;
  sparse_field.data_type = ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32;
  sparse_field.nullable = false;
  sparse_field.dimension = 0;

  TEST_ASSERT(strcmp(sparse_field.name->data, "sparse_field") == 0);
  TEST_ASSERT(sparse_field.data_type == ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32);

  TEST_END();
}

void test_field_helper_functions(void) {
  TEST_START();

  // Test scalar field helper functions
  ZVecInvertIndexParams *invert_params =
      zvec_test_create_default_invert_params(true);
  ZVecFieldSchema *scalar_field = zvec_test_create_scalar_field(
      "test_scalar", ZVEC_DATA_TYPE_INT32, true, invert_params);
  TEST_ASSERT(scalar_field != NULL);
  if (scalar_field) {
    TEST_ASSERT(strcmp(scalar_field->name->data, "test_scalar") == 0);
    TEST_ASSERT(scalar_field->data_type == ZVEC_DATA_TYPE_INT32);
    free(scalar_field);
  }
  if (invert_params) {
    free(invert_params);
  }

  // Test vector field helper functions
  ZVecHnswIndexParams *hnsw_params = zvec_test_create_default_hnsw_params();
  ZVecFieldSchema *vector_field = zvec_test_create_vector_field(
      "test_vector", ZVEC_DATA_TYPE_VECTOR_FP32, 128, false, hnsw_params);
  TEST_ASSERT(vector_field != NULL);
  if (vector_field) {
    TEST_ASSERT(strcmp(vector_field->name->data, "test_vector") == 0);
    TEST_ASSERT(vector_field->data_type == ZVEC_DATA_TYPE_VECTOR_FP32);
    TEST_ASSERT(vector_field->dimension == 128);
    free(vector_field);
  }
  if (hnsw_params) {
    free(hnsw_params);
  }

  TEST_END();
}

// =============================================================================
// Document-related tests
// =============================================================================

void test_doc_creation(void) {
  TEST_START();

  ZVecCollectionSchema *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    // Test complete document creation
    ZVecDoc *doc = zvec_test_create_doc(1, schema, NULL);
    TEST_ASSERT(doc != NULL);
    if (doc) {
      zvec_doc_destroy(doc);
    }

    // Test null value document creation
    ZVecDoc *null_doc = zvec_test_create_doc_null(2, schema, NULL);
    TEST_ASSERT(null_doc != NULL);
    if (null_doc) {
      zvec_doc_destroy(null_doc);
    }

    zvec_collection_schema_destroy(schema);
  }

  TEST_END();
}

void test_doc_primary_key(void) {
  TEST_START();

  // Test primary key generation
  char *pk = zvec_test_make_pk(12345);
  TEST_ASSERT(pk != NULL);
  if (pk) {
    TEST_ASSERT(strcmp(pk, "pk_12345") == 0);
    free(pk);
  }

  TEST_END();
}

void test_doc_functions(void) {
  TEST_START();

  // Create test document using utility function
  ZVecCollectionSchema *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  ZVecDoc *doc = zvec_doc_create();
  TEST_ASSERT(doc != NULL);

  // Test primary key operations
  zvec_doc_set_pk(doc, "test_doc_complete");
  const char *pk = zvec_doc_get_pk_pointer(doc);
  TEST_ASSERT(pk != NULL);
  TEST_ASSERT(strcmp(pk, "test_doc_complete") == 0);

  // Test document ID and score operations
  zvec_doc_set_doc_id(doc, 99999);
  uint64_t doc_id = zvec_doc_get_doc_id(doc);
  TEST_ASSERT(doc_id == 99999);

  zvec_doc_set_score(doc, 0.95f);
  float score = zvec_doc_get_score(doc);
  TEST_ASSERT(score == 0.95f);

  // Test operator operations
  zvec_doc_set_operator(doc, ZVEC_DOC_OP_INSERT);
  ZVecDocOperator op = zvec_doc_get_operator(doc);
  TEST_ASSERT(op == ZVEC_DOC_OP_INSERT);

  ZVecErrorCode err;

  // ==================== COMPREHENSIVE DATA TYPE TESTING ====================

  printf(
      "=== Testing zvec_doc_get_field_value_basic with all supported types "
      "===\n");

  // Test all basic numeric types that zvec_doc_get_field_value_basic supports
  // BOOL type
  ZVecDocField bool_field;
  bool_field.name.data = "bool_field";
  bool_field.name.length = strlen("bool_field");
  bool_field.data_type = ZVEC_DATA_TYPE_BOOL;
  bool_field.value.bool_value = true;
  err = zvec_doc_add_field_by_struct(doc, &bool_field);
  TEST_ASSERT(err == ZVEC_OK);

  bool bool_result;
  err = zvec_doc_get_field_value_basic(doc, "bool_field", ZVEC_DATA_TYPE_BOOL,
                                       &bool_result, sizeof(bool_result));
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(bool_result == true);

  // INT32 type
  ZVecDocField int32_field;
  int32_field.name.data = "int32_field";
  int32_field.name.length = strlen("int32_field");
  int32_field.data_type = ZVEC_DATA_TYPE_INT32;
  int32_field.value.int32_value = -2147483648;  // Min int32
  err = zvec_doc_add_field_by_struct(doc, &int32_field);
  TEST_ASSERT(err == ZVEC_OK);

  int32_t int32_result;
  err = zvec_doc_get_field_value_basic(doc, "int32_field", ZVEC_DATA_TYPE_INT32,
                                       &int32_result, sizeof(int32_result));
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(int32_result == -2147483648);

  // INT64 type
  ZVecDocField int64_field;
  int64_field.name.data = "int64_field";
  int64_field.name.length = strlen("int64_field");
  int64_field.data_type = ZVEC_DATA_TYPE_INT64;
  int64_field.value.int64_value = 9223372036854775807LL;  // Max int64
  err = zvec_doc_add_field_by_struct(doc, &int64_field);
  TEST_ASSERT(err == ZVEC_OK);

  int64_t int64_result;
  err = zvec_doc_get_field_value_basic(doc, "int64_field", ZVEC_DATA_TYPE_INT64,
                                       &int64_result, sizeof(int64_result));
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(int64_result == 9223372036854775807LL);

  // UINT32 type
  ZVecDocField uint32_field;
  uint32_field.name.data = "uint32_field";
  uint32_field.name.length = strlen("uint32_field");
  uint32_field.data_type = ZVEC_DATA_TYPE_UINT32;
  uint32_field.value.uint32_value = 4294967295U;  // Max uint32
  err = zvec_doc_add_field_by_struct(doc, &uint32_field);
  TEST_ASSERT(err == ZVEC_OK);

  uint32_t uint32_result;
  err =
      zvec_doc_get_field_value_basic(doc, "uint32_field", ZVEC_DATA_TYPE_UINT32,
                                     &uint32_result, sizeof(uint32_result));
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(uint32_result == 4294967295U);

  // UINT64 type
  ZVecDocField uint64_field;
  uint64_field.name.data = "uint64_field";
  uint64_field.name.length = strlen("uint64_field");
  uint64_field.data_type = ZVEC_DATA_TYPE_UINT64;
  uint64_field.value.uint64_value = 18446744073709551615ULL;  // Max uint64
  err = zvec_doc_add_field_by_struct(doc, &uint64_field);
  TEST_ASSERT(err == ZVEC_OK);

  uint64_t uint64_result;
  err =
      zvec_doc_get_field_value_basic(doc, "uint64_field", ZVEC_DATA_TYPE_UINT64,
                                     &uint64_result, sizeof(uint64_result));
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(uint64_result == 18446744073709551615ULL);

  // FLOAT type
  ZVecDocField float_field;
  float_field.name.data = "float_field";
  float_field.name.length = strlen("float_field");
  float_field.data_type = ZVEC_DATA_TYPE_FLOAT;
  float_field.value.float_value = 3.14159265359f;
  err = zvec_doc_add_field_by_struct(doc, &float_field);
  TEST_ASSERT(err == ZVEC_OK);

  float float_result;
  err = zvec_doc_get_field_value_basic(doc, "float_field", ZVEC_DATA_TYPE_FLOAT,
                                       &float_result, sizeof(float_result));
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(fabsf(float_result - 3.14159265359f) < 1e-6f);

  // DOUBLE type
  ZVecDocField double_field;
  double_field.name.data = "double_field";
  double_field.name.length = strlen("double_field");
  double_field.data_type = ZVEC_DATA_TYPE_DOUBLE;
  double_field.value.double_value = 2.71828182845904523536;
  err = zvec_doc_add_field_by_struct(doc, &double_field);
  TEST_ASSERT(err == ZVEC_OK);

  double double_result;
  err =
      zvec_doc_get_field_value_basic(doc, "double_field", ZVEC_DATA_TYPE_DOUBLE,
                                     &double_result, sizeof(double_result));
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(fabs(double_result - 2.71828182845904523536) < 1e-15);

  printf(
      "=== Testing zvec_doc_get_field_value_copy with all supported types "
      "===\n");

  // Test STRING type with zvec_doc_get_field_value_copy
  ZVecDocField string_field;
  string_field.name.data = "string_field";
  string_field.name.length = strlen("string_field");
  string_field.data_type = ZVEC_DATA_TYPE_STRING;
  string_field.value.string_value = *zvec_string_create("Hello, 世界!");
  err = zvec_doc_add_field_by_struct(doc, &string_field);
  TEST_ASSERT(err == ZVEC_OK);

  void *string_result;
  size_t string_size;
  err = zvec_doc_get_field_value_copy(
      doc, "string_field", ZVEC_DATA_TYPE_STRING, &string_result, &string_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(string_result != NULL);
  TEST_ASSERT(string_size == strlen("Hello, 世界!"));
  TEST_ASSERT(memcmp(string_result, "Hello, 世界!", string_size) == 0);
  free(string_result);

  // Test BINARY type with zvec_doc_get_field_value_copy
  ZVecDocField binary_field;
  binary_field.name.data = "binary_field";
  binary_field.name.length = strlen("binary_field");
  binary_field.data_type = ZVEC_DATA_TYPE_BINARY;
  uint8_t binary_data[] = {0x00, 0x01, 0x02, 0xFF, 0xFE, 0xFD};
  binary_field.value.string_value =
      *zvec_bin_create(binary_data, sizeof(binary_data));
  err = zvec_doc_add_field_by_struct(doc, &binary_field);
  TEST_ASSERT(err == ZVEC_OK);

  void *binary_result;
  size_t binary_size;
  err = zvec_doc_get_field_value_copy(
      doc, "binary_field", ZVEC_DATA_TYPE_BINARY, &binary_result, &binary_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(binary_result != NULL);
  TEST_ASSERT(binary_size == 6);
  TEST_ASSERT(memcmp(binary_result, "\x00\x01\x02\xFF\xFE\xFD", binary_size) ==
              0);
  free(binary_result);

  // Test VECTOR_FP32 type with zvec_doc_get_field_value_copy
  float test_vector[] = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
  ZVecDocField fp32_vec_field;
  fp32_vec_field.name.data = "fp32_vec_field";
  fp32_vec_field.name.length = strlen("fp32_vec_field");
  fp32_vec_field.data_type = ZVEC_DATA_TYPE_VECTOR_FP32;
  fp32_vec_field.value.vector_value.data = test_vector;
  fp32_vec_field.value.vector_value.length = 5;
  err = zvec_doc_add_field_by_struct(doc, &fp32_vec_field);
  TEST_ASSERT(err == ZVEC_OK);

  void *fp32_vec_result;
  size_t fp32_vec_size;
  err = zvec_doc_get_field_value_copy(doc, "fp32_vec_field",
                                      ZVEC_DATA_TYPE_VECTOR_FP32,
                                      &fp32_vec_result, &fp32_vec_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(fp32_vec_result != NULL);
  TEST_ASSERT(fp32_vec_size == 5 * sizeof(float));
  TEST_ASSERT(memcmp(fp32_vec_result, test_vector, fp32_vec_size) == 0);
  free(fp32_vec_result);


  printf(
      "=== Testing zvec_doc_get_field_value_pointer with all supported types "
      "===\n");

  // Test pointer access to basic types
  const void *bool_ptr;
  size_t bool_ptr_size;
  err = zvec_doc_get_field_value_pointer(doc, "bool_field", ZVEC_DATA_TYPE_BOOL,
                                         &bool_ptr, &bool_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(bool_ptr != NULL);
  TEST_ASSERT(bool_ptr_size == sizeof(bool));
  TEST_ASSERT(*(const bool *)bool_ptr == true);

  const void *int32_ptr;
  size_t int32_ptr_size;
  err = zvec_doc_get_field_value_pointer(
      doc, "int32_field", ZVEC_DATA_TYPE_INT32, &int32_ptr, &int32_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(int32_ptr != NULL);
  TEST_ASSERT(int32_ptr_size == sizeof(int32_t));
  TEST_ASSERT(*(const int32_t *)int32_ptr == -2147483648);

  // Test pointer access to STRING (should return null-terminated C string)
  const void *string_ptr;
  size_t string_ptr_size;
  err = zvec_doc_get_field_value_pointer(doc, "string_field",
                                         ZVEC_DATA_TYPE_STRING, &string_ptr,
                                         &string_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(string_ptr != NULL);
  TEST_ASSERT(string_ptr_size == strlen("Hello, 世界!"));
  TEST_ASSERT(memcmp(string_ptr, "Hello, 世界!", string_ptr_size) == 0);

  // Test pointer access to BINARY
  const void *binary_ptr;
  size_t binary_ptr_size;
  err = zvec_doc_get_field_value_pointer(doc, "binary_field",
                                         ZVEC_DATA_TYPE_BINARY, &binary_ptr,
                                         &binary_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(binary_ptr != NULL);
  TEST_ASSERT(binary_ptr_size == 6);
  TEST_ASSERT(memcmp(binary_ptr, "\x00\x01\x02\xFF\xFE\xFD", binary_ptr_size) ==
              0);

  // Test pointer access to VECTOR_FP32
  const void *fp32_vec_ptr;
  size_t fp32_vec_ptr_size;
  err = zvec_doc_get_field_value_pointer(doc, "fp32_vec_field",
                                         ZVEC_DATA_TYPE_VECTOR_FP32,
                                         &fp32_vec_ptr, &fp32_vec_ptr_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(fp32_vec_ptr != NULL);
  TEST_ASSERT(fp32_vec_ptr_size == 5 * sizeof(float));
  TEST_ASSERT(memcmp(fp32_vec_ptr, test_vector, fp32_vec_ptr_size) == 0);

  // Declare dummy variables for error testing
  const void *dummy_ptr;
  size_t dummy_ptr_size;

  // ==================== FIELD OPERATIONS TESTING ====================

  // Test field operations
  size_t field_count = zvec_doc_get_field_count(doc);
  TEST_ASSERT(field_count >= 10);  // All the fields we've added

  // Test field existence checks
  TEST_ASSERT(zvec_doc_has_field(doc, "bool_field") == true);
  TEST_ASSERT(zvec_doc_has_field(doc, "int32_field") == true);
  TEST_ASSERT(zvec_doc_has_field(doc, "string_field") == true);
  TEST_ASSERT(zvec_doc_has_field(doc, "nonexistent") == false);

  TEST_ASSERT(zvec_doc_has_field_value(doc, "bool_field") == true);
  TEST_ASSERT(zvec_doc_is_field_null(doc, "bool_field") == false);
  TEST_ASSERT(zvec_doc_is_field_null(doc, "nonexistent") == false);

  // Test field names retrieval
  char **field_names;
  size_t name_count;
  err = zvec_doc_get_field_names(doc, &field_names, &name_count);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(name_count >= 10);
  TEST_ASSERT(field_names != NULL);

  // Verify some expected fields are present
  bool found_key_fields = false;
  for (size_t i = 0; i < name_count; i++) {
    if (strcmp(field_names[i], "bool_field") == 0 ||
        strcmp(field_names[i], "int32_field") == 0 ||
        strcmp(field_names[i], "string_field") == 0) {
      found_key_fields = true;
      break;
    }
  }
  TEST_ASSERT(found_key_fields == true);

  zvec_free_str_array(field_names, name_count);

  // ==================== ERROR CONDITION TESTING ====================

  printf("=== Testing error conditions ===\n");

  // Test non-existent field
  err =
      zvec_doc_get_field_value_basic(doc, "missing_field", ZVEC_DATA_TYPE_INT32,
                                     &int32_result, sizeof(int32_result));
  TEST_ASSERT(err != ZVEC_OK);

  err =
      zvec_doc_get_field_value_copy(doc, "missing_field", ZVEC_DATA_TYPE_STRING,
                                    &string_result, &string_size);
  TEST_ASSERT(err != ZVEC_OK);

  err = zvec_doc_get_field_value_pointer(
      doc, "missing_field", ZVEC_DATA_TYPE_FLOAT, &dummy_ptr, &dummy_ptr_size);
  TEST_ASSERT(err != ZVEC_OK);

  // Test wrong data type access
  err =
      zvec_doc_get_field_value_basic(doc, "string_field", ZVEC_DATA_TYPE_INT32,
                                     &int32_result, sizeof(int32_result));
  TEST_ASSERT(err != ZVEC_OK);

  err = zvec_doc_get_field_value_copy(doc, "int32_field", ZVEC_DATA_TYPE_STRING,
                                      &string_result, &string_size);
  TEST_ASSERT(err != ZVEC_OK);

  err = zvec_doc_get_field_value_pointer(
      doc, "bool_field", ZVEC_DATA_TYPE_FLOAT, &dummy_ptr, &dummy_ptr_size);
  TEST_ASSERT(err != ZVEC_OK);

  // ==================== DOCUMENT SERIALIZATION TESTING ====================

  printf("=== Testing document serialization ===\n");

  uint8_t *serialized_data;
  size_t data_size;
  err = zvec_doc_serialize(doc, &serialized_data, &data_size);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(serialized_data != NULL);
  TEST_ASSERT(data_size > 0);

  ZVecDoc *deserialized_doc;
  err = zvec_doc_deserialize(serialized_data, data_size, &deserialized_doc);
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(deserialized_doc != NULL);

  // Verify deserialized document has same field count
  size_t deserialized_field_count = zvec_doc_get_field_count(deserialized_doc);
  TEST_ASSERT(deserialized_field_count == field_count);

  // Test a field from deserialized document
  int32_t deserialized_int32;
  err = zvec_doc_get_field_value_basic(
      deserialized_doc, "int32_field", ZVEC_DATA_TYPE_INT32,
      &deserialized_int32, sizeof(deserialized_int32));
  TEST_ASSERT(err == ZVEC_OK);
  TEST_ASSERT(deserialized_int32 == -2147483648);

  // ==================== CLEANUP ====================

  zvec_doc_destroy(deserialized_doc);
  zvec_free_uint8_array(serialized_data);
  zvec_free_str(string_field.value.string_value.data);
  zvec_free_str(binary_field.value.string_value.data);
  zvec_doc_destroy(doc);
  zvec_collection_schema_destroy(schema);

  TEST_END();
}

// =============================================================================
// Index parameter tests
// =============================================================================

void test_index_params(void) {
  TEST_START();

  // Test HNSW parameter creation
  ZVecHnswIndexParams *hnsw_params = zvec_test_create_default_hnsw_params();
  TEST_ASSERT(hnsw_params != NULL);
  if (hnsw_params) {
    free(hnsw_params);
  }

  // Test Flat parameter creation
  ZVecFlatIndexParams *flat_params = zvec_test_create_default_flat_params();
  TEST_ASSERT(flat_params != NULL);
  if (flat_params) {
    free(flat_params);
  }

  // Test scalar index parameter creation
  ZVecInvertIndexParams *invert_params =
      zvec_test_create_default_invert_params(true);
  TEST_ASSERT(invert_params != NULL);
  if (invert_params) {
    free(invert_params);
  }

  TEST_END();
}

// =============================================================================
// Memory management tests
// =============================================================================
void test_zvec_string_functions(void) {
  TEST_START();

  // Test string creation and basic operations
  ZVecString *str1 = zvec_string_create("Hello World");
  TEST_ASSERT(str1 != NULL);
  TEST_ASSERT(zvec_string_length(str1) == 11);
  TEST_ASSERT(strcmp(zvec_string_c_str(str1), "Hello World") == 0);

  // Test string copy
  ZVecString *str2 = zvec_string_copy(str1);
  TEST_ASSERT(str2 != NULL);
  TEST_ASSERT(zvec_string_length(str2) == 11);
  TEST_ASSERT(strcmp(zvec_string_c_str(str2), "Hello World") == 0);

  // Test string comparison
  int cmp_result = zvec_string_compare(str1, str2);
  TEST_ASSERT(cmp_result == 0);

  ZVecString *str3 = zvec_string_create("Hello");
  TEST_ASSERT(zvec_string_compare(str1, str3) > 0);

  // Test string creation from view
  ZVecStringView view = {"Hello View", 10};
  ZVecString *str4 = zvec_string_create_from_view(&view);
  TEST_ASSERT(str4 != NULL);
  TEST_ASSERT(zvec_string_length(str4) == 10);
  TEST_ASSERT(strcmp(zvec_string_c_str(str4), "Hello View") == 0);

  // Test string view with embedded null bytes
  char binary_data[] = {'H', 'e', 'l', 'l', 'o', '\0', 'W', 'o', 'r', 'l', 'd'};
  ZVecStringView binary_view = {binary_data, 11};
  ZVecString *str5 = zvec_string_create_from_view(&binary_view);
  TEST_ASSERT(str5 != NULL);
  TEST_ASSERT(zvec_string_length(str5) == 11);
  // Note: strcmp will stop at first null byte, so we need to compare manually
  TEST_ASSERT(memcmp(zvec_string_c_str(str5), binary_data, 11) == 0);

  // Cleanup
  zvec_free_string(str1);
  zvec_free_string(str2);
  zvec_free_string(str3);
  zvec_free_string(str4);
  zvec_free_string(str5);

  TEST_END();
}

void test_index_params_functions(void) {
  TEST_START();

  // Test base index params
  ZVecBaseIndexParams base_params;
  zvec_index_params_base_init(&base_params, ZVEC_INDEX_TYPE_HNSW);
  TEST_ASSERT(base_params.index_type == ZVEC_INDEX_TYPE_HNSW);

  // Test invert index params
  ZVecInvertIndexParams invert_params;
  zvec_index_params_invert_init(&invert_params, true, false);
  TEST_ASSERT(invert_params.base.index_type == ZVEC_INDEX_TYPE_INVERT);
  TEST_ASSERT(invert_params.enable_range_optimization == true);
  TEST_ASSERT(invert_params.enable_extended_wildcard == false);

  // Test vector index params
  ZVecVectorIndexParams vector_params;
  zvec_index_params_vector_init(&vector_params, ZVEC_INDEX_TYPE_HNSW,
                                ZVEC_METRIC_TYPE_L2,
                                ZVEC_QUANTIZE_TYPE_UNDEFINED);
  TEST_ASSERT(vector_params.base.index_type == ZVEC_INDEX_TYPE_HNSW);
  TEST_ASSERT(vector_params.metric_type == ZVEC_METRIC_TYPE_L2);
  TEST_ASSERT(vector_params.quantize_type == ZVEC_QUANTIZE_TYPE_UNDEFINED);

  // Test HNSW index params
  ZVecHnswIndexParams hnsw_params;
  zvec_index_params_hnsw_init(&hnsw_params, ZVEC_METRIC_TYPE_COSINE, 16, 200,
                              50, ZVEC_QUANTIZE_TYPE_UNDEFINED);
  TEST_ASSERT(hnsw_params.base.base.index_type == ZVEC_INDEX_TYPE_HNSW);
  TEST_ASSERT(hnsw_params.base.metric_type == ZVEC_METRIC_TYPE_COSINE);
  TEST_ASSERT(hnsw_params.m == 16);
  TEST_ASSERT(hnsw_params.ef_construction == 200);
  TEST_ASSERT(hnsw_params.ef_search == 50);

  // Test Flat index params
  ZVecFlatIndexParams flat_params;
  zvec_index_params_flat_init(&flat_params, ZVEC_METRIC_TYPE_IP,
                              ZVEC_QUANTIZE_TYPE_UNDEFINED);
  TEST_ASSERT(flat_params.base.base.index_type == ZVEC_INDEX_TYPE_FLAT);
  TEST_ASSERT(flat_params.base.metric_type == ZVEC_METRIC_TYPE_IP);

  // Test IVF index params
  ZVecIVFIndexParams ivf_params;
  zvec_index_params_ivf_init(&ivf_params, ZVEC_METRIC_TYPE_L2, 100, 10, true, 5,
                             ZVEC_QUANTIZE_TYPE_UNDEFINED);
  TEST_ASSERT(ivf_params.base.base.index_type == ZVEC_INDEX_TYPE_IVF);
  TEST_ASSERT(ivf_params.n_list == 100);
  TEST_ASSERT(ivf_params.n_iters == 10);
  TEST_ASSERT(ivf_params.use_soar == true);
  TEST_ASSERT(ivf_params.n_probe == 5);

  TEST_END();
}

void test_utility_functions(void) {
  TEST_START();

  // Test error code to string conversion
  const char *error_str = zvec_error_code_to_string(ZVEC_OK);
  TEST_ASSERT(error_str != NULL);
  TEST_ASSERT(strlen(error_str) > 0);

  error_str = zvec_error_code_to_string(ZVEC_ERROR_INVALID_ARGUMENT);
  TEST_ASSERT(error_str != NULL);

  // Test data type to string conversion
  const char *data_type_str = zvec_data_type_to_string(ZVEC_DATA_TYPE_INT32);
  TEST_ASSERT(data_type_str != NULL);
  TEST_ASSERT(strlen(data_type_str) > 0);

  data_type_str = zvec_data_type_to_string(ZVEC_DATA_TYPE_STRING);
  TEST_ASSERT(data_type_str != NULL);

  // Test index type to string conversion
  const char *index_type_str = zvec_index_type_to_string(ZVEC_INDEX_TYPE_HNSW);
  TEST_ASSERT(index_type_str != NULL);
  TEST_ASSERT(strlen(index_type_str) > 0);

  index_type_str = zvec_index_type_to_string(ZVEC_INDEX_TYPE_INVERT);
  TEST_ASSERT(index_type_str != NULL);

  TEST_END();
}

void test_memory_management_functions(void) {
  TEST_START();

  // Test basic memory allocation
  void *ptr = zvec_malloc(1024);
  TEST_ASSERT(ptr != NULL);

  // Test memory reallocation
  void *new_ptr = zvec_realloc(ptr, 2048);
  TEST_ASSERT(new_ptr != NULL);

  // Test memory deallocation
  zvec_free(new_ptr);

  // Test string allocation and deallocation
  ZVecString *str = zvec_string_create("Test String");
  TEST_ASSERT(str != NULL);
  zvec_free_string(str);

  TEST_END();
}

void test_query_params_functions(void) {
  TEST_START();

  // Test basic query parameters creation and destruction
  ZVecQueryParams *base_params = zvec_query_params_create(ZVEC_INDEX_TYPE_HNSW);
  TEST_ASSERT(base_params != NULL);

  // Test union query parameters
  ZVecQueryParamsUnion *union_params =
      zvec_query_params_union_create(ZVEC_INDEX_TYPE_HNSW);
  TEST_ASSERT(union_params != NULL);

  // Test HNSW query parameters
  ZVecHnswQueryParams *hnsw_params = zvec_query_params_hnsw_create(
      ZVEC_INDEX_TYPE_HNSW, 50, 0.5f, false, true);
  TEST_ASSERT(hnsw_params != NULL);

  // Test IVF query parameters
  ZVecIVFQueryParams *ivf_params =
      zvec_query_params_ivf_create(ZVEC_INDEX_TYPE_IVF, 10, true, 1.5f);
  TEST_ASSERT(ivf_params != NULL);

  // Test Flat query parameters
  ZVecFlatQueryParams *flat_params =
      zvec_query_params_flat_create(ZVEC_INDEX_TYPE_FLAT, false, 2.0f);
  TEST_ASSERT(flat_params != NULL);

  // Test setting various parameters on base query params
  ZVecErrorCode err;

  // Test index type setting
  err = zvec_query_params_set_index_type(base_params, ZVEC_INDEX_TYPE_IVF);
  TEST_ASSERT(err == ZVEC_OK);

  // Test radius setting
  err = zvec_query_params_set_radius(base_params, 0.8f);
  TEST_ASSERT(err == ZVEC_OK);

  // Test linear search setting
  err = zvec_query_params_set_is_linear(base_params, false);
  TEST_ASSERT(err == ZVEC_OK);

  // Test refiner setting
  err = zvec_query_params_set_is_using_refiner(base_params, true);
  TEST_ASSERT(err == ZVEC_OK);

  // Test HNSW-specific parameters
  err = zvec_query_params_hnsw_set_ef(hnsw_params, 75);
  TEST_ASSERT(err == ZVEC_OK);

  // Test IVF-specific parameters
  err = zvec_query_params_ivf_set_nprobe(ivf_params, 15);
  TEST_ASSERT(err == ZVEC_OK);

  // Test IVF scale factor setting
  err = zvec_query_params_ivf_set_scale_factor(ivf_params, 2.5f);
  TEST_ASSERT(err == ZVEC_OK);

  // Test destruction of valid parameters
  zvec_query_params_destroy(base_params);
  zvec_query_params_hnsw_destroy(hnsw_params);
  zvec_query_params_ivf_destroy(ivf_params);
  zvec_query_params_flat_destroy(flat_params);
  zvec_query_params_union_destroy(union_params);


  // Test boundary cases - null pointer handling
  zvec_query_params_hnsw_destroy(NULL);
  zvec_query_params_ivf_destroy(NULL);
  zvec_query_params_flat_destroy(NULL);
  zvec_query_params_union_destroy(NULL);


  TEST_END();
}

void test_collection_stats_functions(void) {
  TEST_START();

  char temp_dir[] = "/tmp/zvec_test_collection_stats_functions";

  ZVecCollectionSchema *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    ZVecCollection *collection = NULL;
    ZVecErrorCode err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);

    if (collection) {
      ZVecCollectionStats *stats = NULL;

      // Test normal statistics retrieval
      err = zvec_collection_get_stats(collection, &stats);
      TEST_ASSERT(err == ZVEC_OK);

      if (stats) {
        TEST_ASSERT(stats->doc_count == 0);
        zvec_collection_stats_destroy(stats);
      }

      // Test NULL parameters
      err = zvec_collection_get_stats(NULL, &stats);
      TEST_ASSERT(err != ZVEC_OK);

      err = zvec_collection_get_stats(collection, NULL);
      TEST_ASSERT(err != ZVEC_OK);

      // Test statistics destruction boundary cases
      zvec_collection_stats_destroy(NULL);
      zvec_collection_destroy(collection);
    }

    zvec_collection_schema_destroy(schema);
  }

  // Clean up temporary directory
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", temp_dir);
  system(cmd);

  TEST_END();
}

void test_collection_dml_functions(void) {
  TEST_START();

  char temp_dir[] = "/tmp/zvec_test_collection_dml";

  ZVecCollectionSchema *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    ZVecCollection *collection = NULL;
    ZVecErrorCode err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(collection != NULL);

    if (collection) {
      // Test insertion function boundary cases
      size_t success_count, error_count;

      // Test NULL collection
      err = zvec_collection_insert(NULL, NULL, 0, &success_count, &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      // Test NULL document array
      err = zvec_collection_insert(collection, NULL, 0, &success_count,
                                   &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      // Test zero document count
      ZVecDoc *empty_docs[1];
      err = zvec_collection_insert(collection, (const ZVecDoc **)empty_docs, 0,
                                   &success_count, &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      // Test NULL count pointer
      err = zvec_collection_insert(collection, (const ZVecDoc **)empty_docs, 1,
                                   NULL, &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      // Test update function boundary cases
      err = zvec_collection_update(NULL, NULL, 0, &success_count, &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      err = zvec_collection_update(collection, NULL, 0, &success_count,
                                   &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      err = zvec_collection_update(collection, (const ZVecDoc **)empty_docs, 0,
                                   NULL, &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      // Test upsert function boundary cases
      err = zvec_collection_upsert(NULL, NULL, 0, &success_count, &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      err = zvec_collection_upsert(collection, NULL, 0, &success_count,
                                   &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      err = zvec_collection_upsert(collection, (const ZVecDoc **)empty_docs, 0,
                                   NULL, &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      // Test deletion function boundary cases
      const char *pks[1];
      err = zvec_collection_delete(NULL, NULL, 0, &success_count, &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      err = zvec_collection_delete(collection, NULL, 0, &success_count,
                                   &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      err = zvec_collection_delete(collection, pks, 0, NULL, &error_count);
      TEST_ASSERT(err != ZVEC_OK);

      // Test deletion by filter boundary cases
      err = zvec_collection_delete_by_filter(NULL, NULL);
      TEST_ASSERT(err != ZVEC_OK);

      err = zvec_collection_delete_by_filter(collection, NULL);
      TEST_ASSERT(err != ZVEC_OK);

      zvec_collection_destroy(collection);
    }

    zvec_collection_schema_destroy(schema);
  }

  // Clean up temporary directory
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", temp_dir);
  system(cmd);

  TEST_END();
}

// =============================================================================
// Actual Query Execution Tests
// =============================================================================

void test_actual_vector_queries(void) {
  TEST_START();

  char temp_dir[] = "/tmp/zvec_test_actual_queries";

  // Create schema with vector field
  ZVecCollectionSchema *schema = zvec_collection_schema_create("query_test");
  TEST_ASSERT(schema != NULL);

  if (schema) {
    // Add ID field
    ZVecFieldSchema *id_field =
        zvec_field_schema_create("id", ZVEC_DATA_TYPE_INT64, false, 0);
    zvec_collection_schema_add_field(schema, id_field);

    // Add vector field with HNSW index
    ZVecHnswIndexParams *hnsw_params = zvec_index_params_hnsw_create(
        ZVEC_METRIC_TYPE_L2, ZVEC_QUANTIZE_TYPE_UNDEFINED, 16, 100, 50);
    ZVecFieldSchema *vec_field = zvec_field_schema_create(
        "embedding", ZVEC_DATA_TYPE_VECTOR_FP32, false, 4);
    zvec_field_schema_set_hnsw_index(vec_field, hnsw_params);
    zvec_collection_schema_add_field(schema, vec_field);

    ZVecCollection *collection = NULL;
    ZVecErrorCode err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(collection != NULL);

    if (collection) {
      // Insert test documents
      float vec1[] = {1.0f, 0.0f, 0.0f, 0.0f};
      float vec2[] = {0.0f, 1.0f, 0.0f, 0.0f};
      float vec3[] = {0.0f, 0.0f, 1.0f, 0.0f};
      float vec4[] = {0.7f, 0.7f, 0.0f, 0.0f};  // Similar to vec1 and vec2

      ZVecDoc *docs[4];
      for (int i = 0; i < 4; i++) {
        docs[i] = zvec_doc_create();
        zvec_doc_set_pk(docs[i], zvec_test_make_pk(i + 1));
        zvec_doc_add_field_by_value(docs[i], "id", ZVEC_DATA_TYPE_INT64,
                                    &(int64_t){i + 1}, sizeof(int64_t));
      }

      zvec_doc_add_field_by_value(
          docs[0], "embedding", ZVEC_DATA_TYPE_VECTOR_FP32, vec1, sizeof(vec1));
      zvec_doc_add_field_by_value(
          docs[1], "embedding", ZVEC_DATA_TYPE_VECTOR_FP32, vec2, sizeof(vec2));
      zvec_doc_add_field_by_value(
          docs[2], "embedding", ZVEC_DATA_TYPE_VECTOR_FP32, vec3, sizeof(vec3));
      zvec_doc_add_field_by_value(
          docs[3], "embedding", ZVEC_DATA_TYPE_VECTOR_FP32, vec4, sizeof(vec4));

      size_t success_count, error_count;
      err = zvec_collection_insert(collection, (const ZVecDoc **)docs, 4,
                                   &success_count, &error_count);
      TEST_ASSERT(err == ZVEC_OK);
      TEST_ASSERT(success_count == 4);
      TEST_ASSERT(error_count == 0);

      // Flush collection to build index
      zvec_collection_flush(collection);

      // Test 1: Basic vector search
      ZVecVectorQuery query1 = {0};
      query1.field_name = (ZVecString){.data = "embedding", .length = 9};
      query1.query_vector =
          (ZVecByteArray){.data = (uint8_t *)vec1, .length = sizeof(vec1)};
      query1.topk = 3;
      query1.include_vector = true;
      query1.include_doc_id = true;

      ZVecDoc **results = NULL;
      size_t result_count = 0;
      err = zvec_collection_query(collection, &query1, &results, &result_count);
      TEST_ASSERT(err == ZVEC_OK);
      TEST_ASSERT(result_count > 0);
      TEST_ASSERT(results != NULL);

      // First result should be vec1 itself (distance ~0)
      if (result_count > 0) {
        float score = zvec_doc_get_score(results[0]);
        TEST_ASSERT(score < 0.001f);  // Very small distance
      }

      zvec_docs_free(results, result_count);

      // Test 2: Search with filter
      ZVecVectorQuery query2 = query1;
      query2.filter = (ZVecString){.data = "id > 2", .length = 6};

      err = zvec_collection_query(collection, &query2, &results, &result_count);
      TEST_ASSERT(err == ZVEC_OK);

      // Should only return documents with id > 2
      for (size_t i = 0; i < result_count; i++) {
        int64_t id;
        zvec_doc_get_field_value_basic(results[i], "id", ZVEC_DATA_TYPE_INT64,
                                       &id, sizeof(id));
        TEST_ASSERT(id > 2);
      }

      zvec_docs_free(results, result_count);

      // Cleanup documents
      for (int i = 0; i < 4; i++) {
        zvec_doc_destroy(docs[i]);
      }

      zvec_collection_destroy(collection);
    }

    zvec_collection_schema_destroy(schema);
    zvec_index_params_hnsw_destroy(hnsw_params);
  }

  // Clean up
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", temp_dir);
  system(cmd);

  TEST_END();
}

void test_index_creation_and_management(void) {
  TEST_START();

  char temp_dir[] = "/tmp/zvec_test_index_management";

  ZVecCollectionSchema *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    ZVecCollection *collection = NULL;
    ZVecErrorCode err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(collection != NULL);

    if (collection) {
      // Test 1: Create HNSW index
      ZVecHnswIndexParams *hnsw_params = zvec_index_params_hnsw_create(
          ZVEC_METRIC_TYPE_COSINE, ZVEC_QUANTIZE_TYPE_UNDEFINED, 16, 100, 50);
      TEST_ASSERT(hnsw_params != NULL);

      err = zvec_collection_create_hnsw_index(
          collection, &(ZVecString){.data = "dense", .length = 5}, hnsw_params);
      TEST_ASSERT(err == ZVEC_OK);

      // Test 2: Create scalar index
      ZVecInvertIndexParams *invert_params =
          zvec_index_params_invert_create(true, false);
      TEST_ASSERT(invert_params != NULL);

      err = zvec_collection_create_invert_index(
          collection, &(ZVecString){.data = "name", .length = 4},
          invert_params);
      TEST_ASSERT(err == ZVEC_OK);

      // Note: Index statistics and drop functionality not yet implemented in C
      // API These would require zvec_collection_get_index_stats() and
      // zvec_collection_drop_index()

      // Test 3: Optimize collection
      err = zvec_collection_optimize(collection);
      TEST_ASSERT(err == ZVEC_OK);

      zvec_collection_destroy(collection);
      zvec_index_params_hnsw_destroy(hnsw_params);
      zvec_index_params_invert_destroy(invert_params);
    }

    zvec_collection_schema_destroy(schema);
  }

  // Clean up
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", temp_dir);
  system(cmd);

  TEST_END();
}

void test_collection_ddl_operations(void) {
  TEST_START();

  char temp_dir[] = "/tmp/zvec_test_collection_ddl";

  ZVecCollectionSchema *schema = zvec_test_create_temp_schema();
  TEST_ASSERT(schema != NULL);

  if (schema) {
    ZVecCollection *collection = NULL;
    ZVecErrorCode err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);
    TEST_ASSERT(collection != NULL);

    if (collection) {
      // Test 1: Add new field (using schema modification before opening)
      ZVecFieldSchema *new_field =
          zvec_field_schema_create("new_field", ZVEC_DATA_TYPE_STRING, true, 0);
      TEST_ASSERT(new_field != NULL);

      // Note: Runtime field addition not yet implemented in C API
      // This would require zvec_collection_add_field() which is not implemented

      // Test 2: Get collection schema
      ZVecCollectionSchema *retrieved_schema = NULL;
      err = zvec_collection_get_schema(collection, &retrieved_schema);
      TEST_ASSERT(err == ZVEC_OK);
      TEST_ASSERT(retrieved_schema != NULL);

      size_t field_count =
          zvec_collection_schema_get_field_count(retrieved_schema);
      TEST_ASSERT(field_count > 0);

      zvec_collection_schema_destroy(retrieved_schema);

      zvec_collection_destroy(collection);
    }

    zvec_collection_schema_destroy(schema);
  }

  // Clean up
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", temp_dir);
  system(cmd);

  TEST_END();
}

void test_field_ddl_operations(void) {
  TEST_START();

  // Test field schema creation with various configurations
  ZVecFieldSchema *field1 =
      zvec_field_schema_create("test_field1", ZVEC_DATA_TYPE_STRING, false, 0);
  TEST_ASSERT(field1 != NULL);
  TEST_ASSERT(strcmp(field1->name->data, "test_field1") == 0);
  TEST_ASSERT(field1->data_type == ZVEC_DATA_TYPE_STRING);
  TEST_ASSERT(field1->nullable == false);
  TEST_ASSERT(field1->dimension == 0);

  ZVecFieldSchema *field2 = zvec_field_schema_create(
      "test_field2", ZVEC_DATA_TYPE_VECTOR_FP32, true, 128);
  TEST_ASSERT(field2 != NULL);
  TEST_ASSERT(field2->data_type == ZVEC_DATA_TYPE_VECTOR_FP32);
  TEST_ASSERT(field2->nullable == true);
  TEST_ASSERT(field2->dimension == 128);

  // Test index parameter setting
  ZVecHnswIndexParams *hnsw_params = zvec_index_params_hnsw_create(
      ZVEC_METRIC_TYPE_L2, ZVEC_QUANTIZE_TYPE_UNDEFINED, 16, 100, 50);
  TEST_ASSERT(hnsw_params != NULL);

  ZVecErrorCode err = zvec_field_schema_set_index_params(
      field2, (ZVecIndexParams *)hnsw_params);
  TEST_ASSERT(err == ZVEC_OK);

  // Test field operations
  // (Field validation function doesn't exist in current API)

  // Cleanup
  zvec_field_schema_destroy(field1);
  zvec_field_schema_destroy(field2);
  zvec_index_params_hnsw_destroy(hnsw_params);

  TEST_END();
}

void test_performance_benchmarks(void) {
  TEST_START();

  char temp_dir[] = "/tmp/zvec_test_performance";

  ZVecCollectionSchema *schema = zvec_collection_schema_create("perf_test");
  TEST_ASSERT(schema != NULL);

  if (schema) {
    // Create simple schema for performance testing
    ZVecFieldSchema *id_field =
        zvec_field_schema_create("id", ZVEC_DATA_TYPE_INT64, false, 0);
    zvec_collection_schema_add_field(schema, id_field);

    ZVecFieldSchema *vec_field =
        zvec_field_schema_create("vec", ZVEC_DATA_TYPE_VECTOR_FP32, false, 128);
    ZVecHnswIndexParams *hnsw_params = zvec_index_params_hnsw_create(
        ZVEC_METRIC_TYPE_L2, ZVEC_QUANTIZE_TYPE_UNDEFINED, 16, 100, 50);
    zvec_field_schema_set_hnsw_index(vec_field, hnsw_params);
    zvec_collection_schema_add_field(schema, vec_field);

    ZVecCollection *collection = NULL;
    ZVecErrorCode err =
        zvec_collection_create_and_open(temp_dir, schema, NULL, &collection);
    TEST_ASSERT(err == ZVEC_OK);

    TEST_ASSERT(collection != NULL);

    if (collection) {
      const size_t BATCH_SIZE = 1000;
      const size_t TOTAL_DOCS = 10000;

      // Test bulk insertion performance
#ifdef _POSIX_C_SOURCE
      struct timeval start_time, end_time;
      gettimeofday(&start_time, NULL);
#else
      clock_t start_clock = clock();
#endif

      for (size_t batch_start = 0; batch_start < TOTAL_DOCS;
           batch_start += BATCH_SIZE) {
        ZVecDoc *batch_docs[BATCH_SIZE];
        size_t current_batch_size = (batch_start + BATCH_SIZE > TOTAL_DOCS)
                                        ? TOTAL_DOCS - batch_start
                                        : BATCH_SIZE;

        // Create batch of documents
        for (size_t i = 0; i < current_batch_size; i++) {
          batch_docs[i] = zvec_doc_create();
          zvec_doc_set_pk(batch_docs[i], zvec_test_make_pk(batch_start + i));

          int64_t id = batch_start + i;
          zvec_doc_add_field_by_value(batch_docs[i], "id", ZVEC_DATA_TYPE_INT64,
                                      &id, sizeof(id));

          // Create random vector
          float vec[128];
          for (int j = 0; j < 128; j++) {
            vec[j] = (float)rand() / RAND_MAX;
          }
          zvec_doc_add_field_by_value(batch_docs[i], "vec",
                                      ZVEC_DATA_TYPE_VECTOR_FP32, vec,
                                      sizeof(vec));
        }

        // Insert batch
        size_t success_count, error_count;
        err = zvec_collection_insert(collection, (const ZVecDoc **)batch_docs,
                                     current_batch_size, &success_count,
                                     &error_count);
        TEST_ASSERT(err == ZVEC_OK);
        TEST_ASSERT(success_count == current_batch_size);
        TEST_ASSERT(error_count == 0);

        // Cleanup batch documents
        for (size_t i = 0; i < current_batch_size; i++) {
          zvec_doc_destroy(batch_docs[i]);
        }
      }

#ifdef _POSIX_C_SOURCE
      gettimeofday(&end_time, NULL);
      double insert_time = (end_time.tv_sec - start_time.tv_sec) +
                           (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
#else
      clock_t end_clock = clock();
      double insert_time = ((double)(end_clock - start_clock)) / CLOCKS_PER_SEC;
#endif
      printf("  Inserted %zu documents in %.3f seconds (%.0f docs/sec)\n",
             TOTAL_DOCS, insert_time, TOTAL_DOCS / insert_time);

      // Flush and optimize
      zvec_collection_flush(collection);
      zvec_collection_optimize(collection);

      // Test query performance
      float query_vec[128];
      for (int i = 0; i < 128; i++) {
        query_vec[i] = (float)rand() / RAND_MAX;
      }

      ZVecVectorQuery query = {0};
      query.field_name = (ZVecString){.data = "vec", .length = 3};
      query.query_vector = (ZVecByteArray){.data = (uint8_t *)query_vec,
                                           .length = sizeof(query_vec)};
      query.topk = 10;
      query.include_vector = false;
      query.include_doc_id = true;

      const int QUERY_COUNT = 100;
#ifdef _POSIX_C_SOURCE
      struct timeval query_start_time, query_end_time;
      gettimeofday(&query_start_time, NULL);
#else
      clock_t query_start_clock = clock();
#endif

      for (int q = 0; q < QUERY_COUNT; q++) {
        ZVecDoc **results = NULL;
        size_t result_count = 0;

        err =
            zvec_collection_query(collection, &query, &results, &result_count);
        TEST_ASSERT(err == ZVEC_OK);
        TEST_ASSERT(result_count <= 10);

        zvec_docs_free(results, result_count);
      }

#ifdef _POSIX_C_SOURCE
      gettimeofday(&query_end_time, NULL);
      double query_time =
          (query_end_time.tv_sec - query_start_time.tv_sec) +
          (query_end_time.tv_usec - query_start_time.tv_usec) / 1000000.0;
#else
      clock_t query_end_clock = clock();
      double query_time =
          ((double)(query_end_clock - query_start_clock)) / CLOCKS_PER_SEC;
#endif
      double avg_query_time =
          (query_time * 1000) / QUERY_COUNT;  // ms per query
      printf("  Average query time: %.2f ms\n", avg_query_time);

      zvec_collection_destroy(collection);
      zvec_index_params_hnsw_destroy(hnsw_params);
    }

    zvec_collection_schema_destroy(schema);
  }

  // Clean up
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", temp_dir);
  system(cmd);

  TEST_END();
}

// =============================================================================
// Main function
// =============================================================================

int main(void) {
  printf("Starting comprehensive C API tests...\n\n");

  // Clean up previous test directories
  printf("Cleaning up previous test directories...\n");
  system("rm -rf /tmp/zvec_test_*");
  printf("Cleanup completed.\n\n");

  test_version_functions();
  test_error_handling_functions();
  test_zvec_config();
  test_zvec_initialize();
  test_zvec_string_functions();

  // Schema-related tests
  test_schema_basic_operations();
  test_schema_edge_cases();
  test_schema_field_operations();
  test_normal_schema_creation();
  test_schema_with_indexes();
  test_schema_max_doc_count();

  // Field-related tests
  test_field_schema_functions();
  test_field_helper_functions();
  test_field_ddl_operations();

  // Collection-related tests
  test_collection_basic_operations();
  test_collection_edge_cases();
  test_collection_delete_by_filter();
  test_collection_stats();
  test_collection_stats_functions();
  test_collection_dml_functions();
  test_collection_ddl_operations();

  // Doc-related tests
  test_doc_creation();
  test_doc_primary_key();
  test_doc_functions();

  // Index tests
  test_index_params();
  test_index_params_functions();
  test_index_creation_and_management();

  // Query tests
  test_query_params_functions();
  test_actual_vector_queries();

  // Performance tests
  // test_performance_benchmarks();

  // Utility function tests
  test_utility_functions();

  // Memory management tests
  test_memory_management_functions();

  printf("\n=== Comprehensive Test Summary ===\n");
  printf("Total tests: %d\n", test_count);
  printf("Passed: %d\n", passed_count);
  printf("Failed: %d\n", test_count - passed_count);

  return test_count == passed_count ? 0 : 1;
}
