import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const sourcePath = "/Users/wangyuhang/Desktop/school/王宇航_技术面试汇报_初稿.pptx";
const outputDir = "/Users/wangyuhang/Desktop/school/Litho_cpp/.codex-pptx-edit/work";
const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePath));
const snapshot = await presentation.inspect({
    kind: "deck,slide,textbox,shape,image,layout",
    include: "id,slide,name,title,text,textPreview,bbox,bboxUnit,isPlaceholder,alt,placeholders",
    maxChars: 30000,
});
await fs.writeFile(`${outputDir}/inspection.ndjson`, snapshot.ndjson);
await fs.writeFile(`${outputDir}/slide-5.layout.json`, await (await presentation.slides.getItem(4).export({format: "layout"})).text());
const preview = await presentation.slides.getItem(4).export({format: "png", scale: 2});
await fs.writeFile(`${outputDir}/slide-5.png`, new Uint8Array(await preview.arrayBuffer()));
console.log(`slides=${presentation.slides.items.length}`);
console.log(`masters=${presentation.masters.items.length}`);
console.log(`layouts=${presentation.layouts.items.length}`);
console.log(presentation.masters.items.map((master) => ({id: master.id, name: master.name})));
console.log(presentation.layouts.items.map((layout) => ({id: layout.id, name: layout.name, placeholders: layout.placeholders.summary()})));
for (const id of ["sh/sb6xsvu9", "sh/yhkbe1o7", "sh/z2tcnm5s", "sh/cf2tcr61", "sh/dgbulwnm"]) {
    const item = presentation.resolve(id);
    console.log(id, {
        position: {left: item.left, top: item.top, width: item.width, height: item.height},
        fill: item.fill?.toJSON?.() ?? String(item.fill),
        line: item.line?.toJSON?.() ?? String(item.line),
        text: item.text?.toString?.(),
        textStyle: item.text ? {
            fontSize: item.text.fontSize,
            typeface: item.text.typeface,
            color: item.text.color,
            bold: item.text.bold,
            alignment: item.text.alignment,
            verticalAlignment: item.text.verticalAlignment,
            autoFit: item.text.autoFit,
            insets: item.text.insets,
        } : undefined,
    });
}
