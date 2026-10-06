import fs from "node:fs/promises";
import path from "node:path";
import {pathToFileURL} from "node:url";
import {FileBlob, PresentationFile} from "@oai/artifact-tool";

const sourcePath = "/Users/wangyuhang/Desktop/school/王宇航_技术面试汇报_初稿.pptx";
const workspaceDir = "/Users/wangyuhang/Desktop/school/Litho_cpp";
const skillDir = "/Users/wangyuhang/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations";
const stagingDir = path.join(workspaceDir, ".codex-pptx-edit/finalizer");
const finalPath = path.join(workspaceDir, "presentation-output/王宇航_技术面试汇报_SRAF完善版.pptx");
const runtimePython = "/Users/wangyuhang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const visualsDir = path.join(workspaceDir, ".codex-pptx-edit/sraf-visuals");

const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePath));
const sourceSlide = presentation.slides.getItem(4);
const oldSlide6Number = presentation.resolve("sh/zi98nu94");
const oldSlide7Number = presentation.resolve("sh/dcbud0ra");
oldSlide6Number.text = "07";
oldSlide7Number.text = "08";

const slide = presentation.slides.insert({after: sourceSlide, layoutId: "/ppt/slideLayouts/slideLayout8.xml"}).slide;
slide.shapes.add({geometry: "rect", name: "slide-background", position: {left: 0, top: 0, width: 1280, height: 720}, fill: "#FBFCFE", line: {fill: "none", width: 0}});

function addText(name, text, position, options = {}) {
    const shape = slide.shapes.add({geometry: "textbox", name, position, fill: "none", line: {fill: "none", width: 0}});
    shape.text = text;
    shape.text.style = {
        typeface: "PingFang SC",
        fontSize: options.fontSize ?? 20,
        color: options.color ?? "#102238",
        bold: options.bold ?? false,
        alignment: options.alignment ?? "left",
        verticalAlignment: options.verticalAlignment ?? "top",
        autoFit: "none",
        wrap: "square",
        insets: {top: 0, right: 0, bottom: 0, left: 0},
    };
    return shape;
}

addText("slide-title", "SRAF骨架与控制点提取", {left: 54, top: 42, width: 1110, height: 58}, {fontSize: 42, bold: true});
slide.shapes.add({geometry: "rect", name: "title-accent", position: {left: 54, top: 112, width: 58, height: 5}, fill: "#3A9BD1", line: {fill: "none", width: 0}});
addText("method-summary", "Zhang-Suen细化保留SRAF拓扑，骨架按连接关系排序后间隔取点。", {left: 56, top: 136, width: 1166, height: 42}, {fontSize: 23, bold: true});

const stages = [
    {
        number: "01",
        title: "SRAF二值区域",
        detail: "从LSM结果中分离辅助图形",
        imagePath: path.join(workspaceDir, "result/SRAF_result/initial_sraf_mask.png"),
        alt: "从LSM结果中分离出的SRAF二值区域",
    },
    {
        number: "02",
        title: "一像素骨架",
        detail: "Zhang-Suen细化得到818个骨架像素",
        imagePath: path.join(visualsDir, "sraf_skeleton.png"),
        alt: "Zhang-Suen细化得到的一像素宽SRAF骨架",
    },
    {
        number: "03",
        title: "骨架控制点",
        detail: "每10个骨架像素取点\n分叉结构按图边分别采样",
        imagePath: path.join(visualsDir, "sraf_skeleton_overlay.png"),
        alt: "SRAF骨架、分叉边和绿色控制点叠加图",
    },
    {
        number: "04",
        title: "曲线与宽度重建",
        detail: "中心曲线结合半宽3.1 px\n采用4×4 MSAA完成渲染",
        imagePath: path.join(visualsDir, "sraf_reconstructed.png"),
        alt: "根据中心曲线和优化宽度重建的SRAF",
    },
];
const stageLefts = [54, 352, 650, 948];
for (let index = 0; index < stages.length; ++index) {
    const stage = stages[index];
    const left = stageLefts[index];
    slide.shapes.add({
        geometry: "rect",
        name: `image-frame-${index + 1}`,
        position: {left, top: 202, width: 240, height: 240},
        fill: "#FFFFFF",
        line: {style: "solid", fill: "#D6E1E8", width: 1},
    });
    slide.images.add({
        blob: await fs.readFile(stage.imagePath),
        contentType: "image/png",
        alt: stage.alt,
        fit: "contain",
        position: {left: left + 1, top: 203, width: 238, height: 238},
    });
    addText(`stage-number-${index + 1}`, stage.number, {left, top: 468, width: 34, height: 30}, {fontSize: 20, color: "#3A9BD1", bold: true});
    addText(`stage-title-${index + 1}`, stage.title, {left: left + 38, top: 466, width: 202, height: 34}, {fontSize: 22, bold: true});
    addText(`stage-detail-${index + 1}`, stage.detail, {left, top: 510, width: 240, height: 62}, {fontSize: 18, color: "#718096"});
}

const takeaway = slide.shapes.add({
    geometry: "roundRect",
    name: "representation-summary",
    position: {left: 54, top: 590, width: 1170, height: 44},
    fill: "#EAF4FA",
    line: {fill: "none", width: 0},
    borderRadius: 10,
});
takeaway.text = "当前版图的10个SRAF分量压缩为110个控制点与1个共享宽度变量";
takeaway.text.style = {
    typeface: "PingFang SC",
    fontSize: 19,
    color: "#155C93",
    bold: true,
    alignment: "center",
    verticalAlignment: "middle",
    autoFit: "none",
    insets: {top: 0, right: 12, bottom: 0, left: 12},
};

addText("footer", "王宇航｜技术面试汇报", {left: 54, top: 650, width: 500, height: 24}, {fontSize: 14, color: "#718096"});
addText("page-number", "06", {left: 1170, top: 650, width: 54, height: 24}, {fontSize: 15, color: "#718096", alignment: "right"});
slide.speakerNotes.textFrame.setText([
    "图像由当前C++ demo_sraf_geometry基于result/LSM_result/LSM_mask.txt生成。",
    "Zhang-Suen细化：10个SRAF连通分量，818个骨架像素。",
    "控制点间隔为10个骨架像素，共110个控制点。",
    "曲线重建采用3.1 px半宽和4×4 MSAA。",
].join("\n"));

await fs.mkdir(stagingDir, {recursive: true});
await fs.mkdir(path.dirname(finalPath), {recursive: true});
const candidatePath = path.join(stagingDir, "candidate-sraf-enhanced.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const {finalizePresentation} = await import(pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href);
const result = await finalizePresentation({
    explicitTotalSlideCount: 8,
    requiredNativeTableOwnerSlides: [],
    requiredNativeChartOwnerSlides: [],
    workspaceDir,
    candidatePath,
    finalPath,
    pythonExecutable: runtimePython,
    integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
    layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
    layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit"],
    fontPolicy: {
        basis: "reference",
        families: ["Helvetica Neue", "PingFang SC"],
        referencePath: sourcePath,
        referenceSha256: "15ff7f5eb3342f1dca4f462bbbda93f4815907eb9f11d962731d53b21f2ef689",
    },
    verifyArtifactToolImport: true,
    receiptPath: path.join(stagingDir, "sraf-enhanced.validation.json"),
});
console.log(JSON.stringify(result, null, 2));
