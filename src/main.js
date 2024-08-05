import * as Matrix from "./gl-matrix.js";

const main = async () => {
  /**
   * 🎨 Graphics shaders
   * @uniform pMatrix Projection matrix
   * @uniform vMatrix View matrix
   * @uniform mMatrix Model matrix
   */
  const shader = {
    vertex: `
    struct Uniform {
      pMatrix : mat4x4<f32>,
      vMatrix : mat4x4<f32>,
      mMatrix : mat4x4<f32>,
    };
    @binding(0) @group(0) var<uniform> uniforms : Uniform;

    struct Output {
        @builtin(position) Position : vec4<f32>,
        @location(0) vColor : vec4<f32>,
    };

    @vertex
      fn main(@location(0) pos: vec4<f32>) -> Output {

          var output: Output;
          output.Position = uniforms.pMatrix * uniforms.vMatrix * uniforms.mMatrix * pos;
          output.vColor = vec4<f32>(pos.y + 0.5, 0.0, 1.0 - pos.y, 1.0);

          return output;
      }
  `,

    fragment: `
    @fragment
    fn main(@location(0) vColor: vec4<f32>) -> @location(0) vec4<f32> {
    return vColor;
  }
  `,
  };

  //---------------------------------------------------
  /**
   * 🕸️ Create the basic grid of points (vertices)
   */
  const gridSize = 20;
  const gridVertices = new Float32Array(gridSize * gridSize * 3);
  for (let i = 0; i < gridSize; i++) {
    for (let j = 0; j < gridSize; j++) {
      const index = (i * gridSize + j) * 3; // Индекс для x, y, z
      // Vertex
      gridVertices[index] = (-gridSize * 0.5 + j) * 0.5; // j for x
      gridVertices[index + 1] = 0.0; // y
      gridVertices[index + 2] = (-gridSize * 0.5 + i) * 0.5; // i for z
    }
  }

  const gridVelocity = new Float32Array(gridSize * gridSize * 3);
  for (let i = 0; i < gridVertices.length; i += 3) {
    gridVelocity[i + 0] = 0.0;
    gridVelocity[i + 1] = 0.0;
    gridVelocity[i + 2] = 0.0;
  }

  /**
   * 🧮 Indexing the basic grid of points
   */
  const gridIndexes = new Uint32Array((gridSize - 1) * gridSize * 2); // Используем Uint32Array для индексов
  let index = 0;
  for (let i = 0; i < gridSize - 1; i++) {
    for (let j = 0; j < gridSize; j++) {
      for (let k = 0; k < 2; k++) {
        gridIndexes[index++] = j + gridSize * (i + k); // Заполняем индекс
      }
    }
  }

  //---------------------------------------------------
  /**
   * 🪄 Initialisation
   */
  const canvas = document.getElementById("canvas-webgpu");
  canvas.width = 1200;
  canvas.height = 800;

  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");

  const devicePixelRatio = window.devicePixelRatio || 1;
  const size = [canvas.clientWidth, canvas.clientHeight];

  const format = navigator.gpu.getPreferredCanvasFormat(); // формат данных в которых храняться пиксели в физическом устройстве

  context.configure({
    device: device,
    format: format,
    size: size,
    compositingAlphaMode: "opaque",
  });

  //---------------------------------------------------
  /**
   * 🧾 Create uniform data
   */
  let VIEWMATRIX = glMatrix.mat4.create();
  let PROJMATRIX = glMatrix.mat4.create();
  let MODELMATRIX = glMatrix.mat4.create();

  glMatrix.mat4.lookAt(
    VIEWMATRIX,
    [0.0, 18.0, -18.0],
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0]
  );

  glMatrix.mat4.identity(PROJMATRIX);
  let fovy = (20 * Math.PI) / 180;
  glMatrix.mat4.perspective(
    PROJMATRIX,
    fovy,
    canvas.width / canvas.height,
    1.0,
    100
  );

  //---------------------------------------------------
  /**
   * 🔺 Vertex buffer to GPU
   */
  const vertexBuffer = device.createBuffer({
    size: gridVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, //COPY_DST  ХЗ что это
    mappedAtCreation: true,
  });

  new Float32Array(vertexBuffer.getMappedRange()).set(gridVertices);
  vertexBuffer.unmap();

  //---------------------------------------------------
  /**
   * 🔺 Index buffer to GPU
   */
  const indexBuffer = device.createBuffer({
    size: gridIndexes.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });

  new Uint32Array(indexBuffer.getMappedRange()).set(gridIndexes);
  indexBuffer.unmap();

  //---------------------------------------------------
  /**
   * 🎨 Render pipeline
   */
  const renderPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: device.createShaderModule({
        code: shader.vertex,
      }),
      entryPoint: "main",
      buffers: [
        {
          arrayStride: 4 * 3,
          attributes: [
            {
              shaderLocation: 0,
              format: "float32x3",
              offset: 0,
            },
          ],
        },
      ],
    },
    fragment: {
      module: device.createShaderModule({
        code: shader.fragment,
      }),
      entryPoint: "main",
      targets: [
        {
          format: format,
        },
      ],
    },
    primitive: {
      topology: "line-strip",
      stripIndexFormat: "uint32",
      //cullMode: 'front',
      //topology: "point-list",
    },
    depthStencil: {
      format: "depth24plus",
      depthWriteEnabled: true, // Enable depth test
      depthCompare: "less",
    },
  });

  // create uniform buffer and layout
  const uniformBuffer = device.createBuffer({
    size: 64 + 64 + 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const uniformBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: uniformBuffer,
          offset: 0,
          size: 64 + 64 + 64, // PROJMATRIX + VIEWMATRIX + MODELMATRIX // Каждая матрица занимает 64 байта
        },
      },
    ],
  });

  let textureView = context.getCurrentTexture().createView();

  let depthTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const renderPassDescription = {
    colorAttachments: [
      {
        view: textureView,
        clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
    depthStencilAttachment: {
      view: depthTexture.createView(),
      depthLoadOp: "clear",
      depthClearValue: 1.0,
      depthStoreOp: "store",
    },
  };

  //---------------------------------------------------
  /**
   * 📝 Write uniforms
   */
  device.queue.writeBuffer(uniformBuffer, 0, PROJMATRIX);
  device.queue.writeBuffer(uniformBuffer, 64, VIEWMATRIX);

  //---------------------------------------------------
  /**
   * 🧮 Compute shader
   */
  const moduleСompute = device.createShaderModule({
    label: "compute module",
    code: `
        struct Point {
          pos : vec3<f32>,
        }

        struct Points {
          points : array<Point>,
        }

        @group(0) @binding(0) var<storage, read> input: Points;
        @group(0) @binding(1) var<storage, read_write> output: Points;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>)
        {
          let index: u32 = id.x;

          if (index < u32(arrayLength(&input.points)))
          {
              let point = input.points[index];

              if (index != 0 && index != 14 && index != 210 && index != 224)
              {
                let x = point.pos.x;
                let y = point.pos.y;
                let z = point.pos.z;
              }
              else
              {
                let x = point.pos.x;
                let y = point.pos.y - 0.005;
                let z = point.pos.z;
                output.points[index].pos = vec3<f32>(x, y, z);
              }
          }
        }
        `,
  });

  //---------------------------------------------------
  /**
   * 🪈 Compute pipeline
   */
  const pipelineCompute = device.createComputePipeline({
    label: "compute pipeline",
    layout: "auto",
    compute: {
      module: moduleСompute,
      entryPoint: "main",
    },
  });

  const workBuffer_A = device.createBuffer({
    label: "work buffer A",
    size: gridVertices.byteLength,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(workBuffer_A, 0, gridVertices);

  const workBuffer_B = device.createBuffer({
    label: "work buffer B",
    size: gridVertices.byteLength,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(workBuffer_B, 0, gridVertices);

  // create a buffer on the GPU to get a copy of the results
  const resultBuffer = device.createBuffer({
    label: "result buffer",
    size: gridVertices.byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // Setup a bindGroup to tell the shader which
  // buffer to use for the computation
  const bindGroupCompute_A = device.createBindGroup({
    label: "bindGroup for work buffer A",
    layout: pipelineCompute.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: workBuffer_A } },
      { binding: 1, resource: { buffer: workBuffer_B } },
    ],
  });

  // Setup a bindGroup to tell the shader which
  // buffer to use for the computation
  const bindGroupCompute_B = device.createBindGroup({
    label: "bindGroup for work buffer B",
    layout: pipelineCompute.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: workBuffer_B } },
      { binding: 1, resource: { buffer: workBuffer_A } },
    ],
  });

  const bindGroupRender_A = device.createBindGroup({
    label: "bindGroup for bindGroupRender",
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: workBuffer_B } }],
  });

  const bindGroupRender_B = device.createBindGroup({
    label: "bindGroup for bindGroupRender",
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: workBuffer_A } }],
  });

  const bindGroupsCompute = [
    {
      bindGroup: bindGroupCompute_A,
      buffer: workBuffer_B,
      bindGroupRender: bindGroupRender_A,
    },
    {
      bindGroup: bindGroupCompute_B,
      buffer: workBuffer_A,
      bindGroupRender: bindGroupRender_B,
    },
  ];

  let time_old = 0;
  let sinAdvance = 0; // Advance fo sin func
  let t = 0; // A buffer or B buffer (t % 2)

  // Clone of vertices array
  const gridNewVertices = gridVertices.slice();

  //---------------------------------------------------
  /**
   * 🎰 Main Сycle
   */
  async function mainСycle(time) {
    //-----------------TIME-----------------------------
    let dt = (time - time_old) * 0.00001;
    time_old = time;
    //--------------------------------------------------

    //------------------MATRIX EDIT---------------------
    if (getRotateXMotion()) {
      glMatrix.mat4.rotateX(MODELMATRIX, MODELMATRIX, dt * 100);
    }
    if (getRotateYMotion()) {
      glMatrix.mat4.rotateY(MODELMATRIX, MODELMATRIX, dt * 100);
    }
    if (getRotateZMotion()) {
      glMatrix.mat4.rotateZ(MODELMATRIX, MODELMATRIX, dt * 100);
    }

    document
      .querySelector('input[name="reset"]')
      .addEventListener("click", function () {
        MODELMATRIX = glMatrix.mat4.create(); // переопределение переменной MODELMATRIX
      });

    device.queue.writeBuffer(uniformBuffer, 128, MODELMATRIX); // Model matrix have offset 128

    //Encode commands to do the computation
    const encoder = device.createCommandEncoder({
      label: "doubling encoder",
    });

    const computePass = encoder.beginComputePass({
      label: "doubling compute pass",
    });

    computePass.setPipeline(pipelineCompute);
    computePass.setBindGroup(0, bindGroupsCompute[t % 2].bindGroup);
    computePass.dispatchWorkgroups(Math.ceil(64));
    computePass.end();

    encoder.copyBufferToBuffer(
      bindGroupsCompute[t % 2].buffer,
      0,
      resultBuffer,
      0,
      resultBuffer.size
    );

    device.queue.submit([encoder.finish()]);

    // Read the results
    // await resultBuffer.mapAsync(GPUMapMode.READ);
    // let result = new Float32Array(resultBuffer.getMappedRange().slice());
    // resultBuffer.unmap();

    //---------------------------------------------------
    /**
     * 🔮 Physics
     */
    // NOTE: In future make it in GPU
    const g = -isGravityChecked() * 4.8;
    const k = 50.15;
    const d = 0.009;

    for (let i = 0; i < gridSize * gridSize * 3; i += 3) {
      if (
        i == 0 ||
        i == gridSize * 3 - 3 ||
        i == gridSize * gridSize * 3 - 3 ||
        i == gridSize * (gridSize * 3 - 3)
      ) {
        continue;
      }

      gridVelocity[i + 1] += g; // Force Y + g

      let x = gridNewVertices[i];
      let y = gridNewVertices[i + 1];
      let z = gridNewVertices[i + 2];

      // Hooke's law
      let shx = 0; // summ of Hooke's X
      let shy = 0;
      let shz = 0;
      for (let j1 = -1; j1 <= 1; j1++) {
        for (let j2 = -1; j2 <= 1; j2++) {
          // Пропускаем центральную точку
          if (j1 === 0 && j2 === 0) continue;

          // Индексы соседних точек
          const ix = i + j1 * gridSize * 3 + j2 * 3 + 0;
          const iy = i + j1 * gridSize * 3 + j2 * 3 + 1;
          const iz = i + j1 * gridSize * 3 + j2 * 3 + 2;

          // Проверка на выход за границы массива
          if (ix < 0 || ix >= gridSize * gridSize * 3) continue;

          // Расчет вектора силы натяжения между текущей точкой и соседней
          const lengthX = gridNewVertices[ix] - gridVertices[ix];
          const lengthY = gridNewVertices[iy] - gridVertices[iy];
          const lengthZ = gridNewVertices[iz] - gridVertices[iz];

          // Вычисление силы натяжения
          const restLength = Math.sqrt(
            (x - gridVertices[ix]) ** 2 +
              (y - gridVertices[iy]) ** 2 +
              (z - gridVertices[iz]) ** 2
          );
          const currentLength = Math.sqrt(
            (x - gridNewVertices[ix]) ** 2 +
              (y - gridNewVertices[iy]) ** 2 +
              (z - gridNewVertices[iz]) ** 2
          );
          const distance = Math.sqrt(
            lengthX ** 2 + lengthY ** 2 + lengthZ ** 2
          );

          const forceMagnitude = k * (currentLength - restLength); // Сила по закону Гука

          // Применение силы натяжения
          if (distance != 0) {
            shx += (lengthX / distance) * forceMagnitude;
            shy += (lengthY / distance) * forceMagnitude;
            shz += (lengthZ / distance) * forceMagnitude;
          }
        }
      }

      // Dumping law
      let sdx = 0;
      let sdy = 0;
      let sdz = 0;
      for (let j1 = -1; j1 <= 1; j1++) {
        for (let j2 = -1; j2 <= 1; j2++) {
          // Пропускаем центральную точку
          if (j1 === 0 && j2 === 0) continue;

          // Индексы соседних точек
          const ix = i + j1 * gridSize * 3 + j2 * 3 + 0;
          const iy = i + j1 * gridSize * 3 + j2 * 3 + 1;
          const iz = i + j1 * gridSize * 3 + j2 * 3 + 2;

          // Проверка на выход за границы массива
          if (ix < 0 || ix >= gridSize * gridSize * 3) continue;

          // Применение силы натяжения
          sdx += d * (gridVelocity[i] - gridVelocity[ix]);
          sdy += d * (gridVelocity[i + 1] - gridVelocity[iy]);
          sdz += d * (gridVelocity[i + 2] - gridVelocity[iz]);
        }
      }

      gridVelocity[i] += shx - sdx;
      gridVelocity[i + 1] += shy - sdy;
      gridVelocity[i + 2] += shz - sdz;
    }

    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const index = (i * gridSize + j) * 3; // Индекс для x, y, z
        // Vertex
        gridNewVertices[index] += gridVelocity[index] * dt; // j используется для x
        gridNewVertices[index + 1] += gridVelocity[index + 1] * dt;
        gridNewVertices[index + 2] += gridVelocity[index + 2] * dt; // i используется для z

        if (index == 507 && isSinChecked()) {
          gridNewVertices[index + 1] = Math.sin(sinAdvance * 100) * 5;
        }
      }
    }
    sinAdvance += dt * 10;

    device.queue.writeBuffer(vertexBuffer, 0, gridNewVertices);
    //device.queue.writeBuffer(vertexBuffer, 0, result);

    //---------------------------------------------------
    /**
     * 🎨🎨 Render
     */
    const renderCommandEncoder = device.createCommandEncoder();
    textureView = context.getCurrentTexture().createView();
    renderPassDescription.colorAttachments[0].view = textureView;

    const renderPass = renderCommandEncoder.beginRenderPass(
      renderPassDescription
    );

    renderPass.setPipeline(renderPipeline);
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.setIndexBuffer(indexBuffer, "uint32");
    renderPass.setBindGroup(0, uniformBindGroup);
    renderPass.drawIndexed(gridIndexes.length);
    renderPass.end();

    device.queue.submit([renderCommandEncoder.finish()]);

    t++;
    requestAnimationFrame(mainСycle);
  }

  mainСycle(0);
};

//---------------------------------------------------
/**
 * 🧾 UI check
 */
function getRotateXMotion() {
  const selectedRadio = document.querySelector('input[name="motionX"]');
  return selectedRadio.checked;
}

function getRotateYMotion() {
  const selectedRadio = document.querySelector('input[name="motionY"]');
  return selectedRadio.checked;
}

function getRotateZMotion() {
  const selectedRadio = document.querySelector('input[name="motionZ"]');
  return selectedRadio.checked;
}

function isGravityChecked() {
  const gravityCheckbox = document.querySelector('input[name="gravity"]');
  return gravityCheckbox.checked;
}

function isSinChecked() {
  const gravityCheckbox = document.querySelector('input[name="sin"]');
  return gravityCheckbox.checked;
}

main();
