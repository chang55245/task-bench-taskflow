/* Copyright 2020 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdarg.h>
#include <assert.h>
#include <string.h>
#include <algorithm> 
#include <unistd.h>
#include <vector>
#include <memory>
#include <taskflow/taskflow.hpp>
#include "core.h"
#include "timer.h"

#define VERBOSE_LEVEL 0
#define USE_CORE_VERIFICATION
#define MAX_NUM_ARGS 10

typedef struct tile_s {
  float dep;
  char *output_buff;
} tile_t;

typedef struct payload_s {
  int x;
  int y;
  TaskGraph graph;
} payload_t;

typedef struct task_args_s {
  int x;
  int y;
} task_args_t;

typedef struct matrix_s {
  tile_t *data;
  int M;
  int N;
} matrix_t;

struct TaskflowApp : public App {
  TaskflowApp(int argc, char **argv);
  ~TaskflowApp();
  void execute_main_loop();
  void execute_timestep(size_t idx, long t, tf::Taskflow& taskflow, 
                       std::vector<std::vector<tf::Task>>& task_matrix);
private:
  tf::Task create_task(const task_args_t *args, int num_args, payload_t payload, 
                      size_t graph_id, tf::Taskflow& taskflow);
  void debug_printf(int verbose_level, const char *format, ...);
private:
  int nb_workers;
  matrix_t *matrix;
  std::vector<char*> extra_local_memory;
};

TaskflowApp::TaskflowApp(int argc, char **argv)
  : App(argc, argv)
{ 
  nb_workers = std::thread::hardware_concurrency();
  
  for (int k = 1; k < argc; k++) {
    if (!strcmp(argv[k], "-worker")) {
      nb_workers = atol(argv[++k]);
    }
  }
  
  matrix = (matrix_t *)malloc(sizeof(matrix_t) * graphs.size());
  
  size_t max_scratch_bytes_per_task = 0;
  
  for (unsigned i = 0; i < graphs.size(); i++) {
    TaskGraph &graph = graphs[i];
    
    matrix[i].M = graph.nb_fields;
    matrix[i].N = graph.max_width;
    matrix[i].data = (tile_t*)malloc(sizeof(tile_t) * matrix[i].M * matrix[i].N);
  
    for (int j = 0; j < matrix[i].M * matrix[i].N; j++) {
      matrix[i].data[j].output_buff = (char *)malloc(sizeof(char) * graph.output_bytes_per_task);
    }
    
    if (graph.scratch_bytes_per_task > max_scratch_bytes_per_task) {
      max_scratch_bytes_per_task = graph.scratch_bytes_per_task;
    }
    
    printf("graph id %d, M = %d, N = %d, data %p, nb_fields %d\n", 
           i, matrix[i].M, matrix[i].N, matrix[i].data, graph.nb_fields);
  }
  
  // Allocate scratch memory for each worker
  extra_local_memory.resize(nb_workers);
  for (int k = 0; k < nb_workers; k++) {
    if (max_scratch_bytes_per_task > 0) {
      extra_local_memory[k] = (char*)malloc(sizeof(char) * max_scratch_bytes_per_task);
      TaskGraph::prepare_scratch(extra_local_memory[k], sizeof(char) * max_scratch_bytes_per_task);
    } else {
      extra_local_memory[k] = nullptr;
    }
  }
}

TaskflowApp::~TaskflowApp()
{
  for (unsigned i = 0; i < graphs.size(); i++) {
    for (int j = 0; j < matrix[i].M * matrix[i].N; j++) {
      free(matrix[i].data[j].output_buff);
      matrix[i].data[j].output_buff = nullptr;
    }
    free(matrix[i].data);
    matrix[i].data = nullptr;
  }
  
  free(matrix);
  matrix = nullptr;
  
  for (int j = 0; j < nb_workers; j++) {
    if (extra_local_memory[j] != nullptr) {
      free(extra_local_memory[j]);
      extra_local_memory[j] = nullptr;
    }
  }
}

void TaskflowApp::execute_main_loop()
{ 
  display();
  
  Timer::time_start();
  
  tf::Executor executor(nb_workers);
  tf::Taskflow taskflow;
  
  // Create task matrix to store task references for dependency setup
  std::vector<std::vector<std::vector<tf::Task>>> all_task_matrices(graphs.size());
  
  // Build the complete task graph
  for (unsigned i = 0; i < graphs.size(); i++) {
    const TaskGraph &g = graphs[i];
    all_task_matrices[i].resize(g.timesteps);
    
    for (int t = 0; t < g.timesteps; t++) {
      all_task_matrices[i][t].resize(g.max_width);
      execute_timestep(i, t, taskflow, all_task_matrices[i]);
    }
  }
  
  // Execute the taskflow
  executor.run(taskflow).wait();
  
  double elapsed = Timer::time_end();
  report_timing(elapsed);
}

void TaskflowApp::execute_timestep(size_t idx, long t, tf::Taskflow& taskflow, 
                                  std::vector<std::vector<tf::Task>>& task_matrix)
{
  const TaskGraph &g = graphs[idx];
  long offset = g.offset_at_timestep(t);
  long width = g.width_at_timestep(t);
  long dset = g.dependence_set_at_timestep(t);
  int nb_fields = g.nb_fields;
  
  task_args_t args[MAX_NUM_ARGS];
  payload_t payload;
  int num_args = 0;
  int ct = 0;  
  
  for (int x = offset; x <= offset + width - 1; x++) {
    std::vector<std::pair<long, long>> deps = g.dependencies(dset, x);   
    num_args = 0;
    ct = 0;    
    
    if (deps.size() == 0) {
      num_args = 1;
      debug_printf(1, "%d[%d] ", x, num_args);
      args[ct].x = x;
      args[ct].y = t % nb_fields;
      ct++;
    } else {
      if (t == 0) {
        num_args = 1;
        debug_printf(1, "%d[%d] ", x, num_args);
        args[ct].x = x;
        args[ct].y = t % nb_fields;
        ct++;
      } else {
        num_args = 1;
        args[ct].x = x;
        args[ct].y = t % nb_fields;
        ct++;
        long last_offset = g.offset_at_timestep(t-1);
        long last_width = g.width_at_timestep(t-1);
        for (std::pair<long, long> dep : deps) {
          num_args += dep.second - dep.first + 1;
          debug_printf(1, "%d[%d, %d, %d] ", x, num_args, dep.first, dep.second); 
          for (int i = dep.first; i <= dep.second; i++) {
            if (i >= last_offset && i < last_offset + last_width) {
              args[ct].x = i;
              args[ct].y = (t-1) % nb_fields;
              ct++;
            } else {
              num_args--;
            }
          }
        }
      }
    }
    
    assert(num_args == ct);
    
    payload.y = t;
    payload.x = x;
    payload.graph = g;
    
    // Create the task and store it in the task matrix
    task_matrix[t][x] = create_task(args, num_args, payload, idx, taskflow);
    
    // Set up dependencies with previous timestep tasks
    if (t > 0) {
      for (int arg_idx = 1; arg_idx < num_args; arg_idx++) {
        int dep_x = args[arg_idx].x;
        if (dep_x >= 0 && dep_x < g.max_width) {
          task_matrix[t-1][dep_x].precede(task_matrix[t][x]);
        }
      }
    }
  }
}

tf::Task TaskflowApp::create_task(const task_args_t *args, int num_args, payload_t payload, 
                                 size_t graph_id, tf::Taskflow& taskflow)
{
  tile_t *mat = matrix[graph_id].data;
  int x0 = args[0].x;
  int y0 = args[0].y;
  
  // Create a lambda that captures all necessary data
  auto task_lambda = [this, mat, args, num_args, payload, graph_id, x0, y0]() {
    // Get worker ID for scratch memory (approximation since Taskflow doesn't expose worker ID directly)
    static thread_local int worker_id = 0;
    static std::atomic<int> worker_counter{0};
    if (worker_id == 0) {
      worker_id = worker_counter.fetch_add(1) % nb_workers;
    }
    
#if defined(USE_CORE_VERIFICATION)    
    TaskGraph graph = payload.graph;
    char *output_ptr = mat[y0 * matrix[graph_id].N + x0].output_buff;
    size_t output_bytes = graph.output_bytes_per_task;
    std::vector<const char *> input_ptrs;
    std::vector<size_t> input_bytes;
    
    // Add input dependencies
    for (int i = 1; i < num_args; i++) {
      int xi = args[i].x;
      int yi = args[i].y;
      input_ptrs.push_back(mat[yi * matrix[graph_id].N + xi].output_buff);
      input_bytes.push_back(graph.output_bytes_per_task);
    }
    
    char* scratch_ptr = (extra_local_memory[worker_id] != nullptr) ? 
                       extra_local_memory[worker_id] : nullptr;
    size_t scratch_bytes = graph.scratch_bytes_per_task;
    
    graph.execute_point(payload.y, payload.x, output_ptr, output_bytes,
                       input_ptrs.data(), input_bytes.data(), input_ptrs.size(), 
                       scratch_ptr, scratch_bytes);
#else  
    mat[y0 * matrix[graph_id].N + x0].dep = 0;
    for (int i = 1; i < num_args; i++) {
      int xi = args[i].x;
      int yi = args[i].y;
      mat[y0 * matrix[graph_id].N + x0].dep += mat[yi * matrix[graph_id].N + xi].dep;
    }
    mat[y0 * matrix[graph_id].N + x0].dep += 1;
    printf("Task tid %d, x %d, y %d, out %f, num_args %d\n", 
           worker_id, payload.x, payload.y, 
           mat[y0 * matrix[graph_id].N + x0].dep, num_args);
#endif
  };
  
  return taskflow.emplace(task_lambda);
}

void TaskflowApp::debug_printf(int verbose_level, const char *format, ...)
{
  if (verbose_level > VERBOSE_LEVEL) {
    return;
  }
  va_list args;
  va_start(args, format);
  vprintf(format, args);
  va_end(args);
}

int main(int argc, char ** argv)
{
  // Create a new argv array with the desired arguments
  const char* new_argv[] = {
    "taskflow_main",  // program name
    "-steps", "4",
    "-width", "8", 
    "-type", "tree",
    "-kernel", "compute_bound",
    "-iter", "4096",
    "-worker", "16"
  };
  
  int new_argc = sizeof(new_argv) / sizeof(new_argv[0]);
  
  // Cast away const for compatibility with App constructor
  TaskflowApp app(new_argc, const_cast<char**>(new_argv));
  app.execute_main_loop();
  return 0;
}
